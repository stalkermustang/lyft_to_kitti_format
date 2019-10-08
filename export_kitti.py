# based on
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_kitti.py

import os
from typing import List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from custom_nuscenes import NuScenes


class KittiConverter:
    def __init__(self,
                 nusc_kitti_dir: str = '~/nusc_kitti/training/',
                 lidar_name: str = 'LIDAR_TOP',
                 get_all_detections = False,
                 samples_count: Optional[int] = None,
                 ):
        """
        :param nusc_kitti_dir: Where to write the KITTI-style annotations.
        :param lidar_name: Name of the lidar sensor.
        :param get_all_detections: If True, will write all
            bboxes in PointCloud and use only FrontCamera.
        :param samples_count: Number of samples to convert.
        :param split: Dataset split to use.
        """
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.lidar_name = lidar_name
        self.get_all_detections = get_all_detections
        self.samples_count = samples_count

        # Create nusc_kitti_dir.
        if not os.path.isdir(self.nusc_kitti_dir):
            os.makedirs(self.nusc_kitti_dir)

        # Select subset of the data to look at.
        self.nusc = NuScenes()

    def nuscenes_gt_to_kitti(self) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse

        token_idx = 0  # Start tokens from 0.

        # Get assignment of scenes to splits.
        split_logs = [
            self.nusc.get('log', scene['log_token'])['logfile']
            for scene
            in self.nusc.scene
        ]

        # Create output folders.
        label_folder = os.path.join(self.nusc_kitti_dir, 'label_2')
        calib_folder = os.path.join(self.nusc_kitti_dir, 'calib')
        image_folder = os.path.join(self.nusc_kitti_dir, 'image_2')
        lidar_folder = os.path.join(self.nusc_kitti_dir, 'velodyne')
        for folder in [label_folder, calib_folder, image_folder, lidar_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        if self.samples_count is not None:
            sample_tokens = sample_tokens[:self.samples_count]
        
        if self.get_all_detections:
            cams_to_see = ['CAM_FRONT']
        else:
            cams_to_see = [
                'CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT'
            ]

        for sample_token in tqdm(sample_tokens):
            # Get sample data.
            sample = self.nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']


            lidar_token = sample['data'][self.lidar_name]
            sd_record_lid = self.nusc.get('sample_data', lidar_token)
            cs_record_lid = self.nusc.get(
                'calibrated_sensor', sd_record_lid['calibrated_sensor_token'])
            for cam_name in cams_to_see:
                cam_front_token = sample['data'][cam_name]
                token_to_write = cam_front_token

                # Retrieve sensor records.
                sd_record_cam = self.nusc.get('sample_data', cam_front_token)
                cs_record_cam = self.nusc.get(
                    'calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
                cam_height = sd_record_cam['height']
                cam_width = sd_record_cam['width']
                imsize = (cam_width, cam_height)

                # Combine transformations and convert to KITTI format.
                # Note: cam uses same conventions in KITTI and nuScenes.
                lid_to_ego = transform_matrix(
                    cs_record_lid['translation'],
                    Quaternion(cs_record_lid['rotation']),
                    inverse=False
                )
                ego_to_cam = transform_matrix(
                    cs_record_cam['translation'],
                    Quaternion(cs_record_cam['rotation']),
                    inverse=True
                )
                velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

                # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
                velo_to_cam_kitti = np.dot(
                    velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

                # Currently not used.
                imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
                r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

                # Projection matrix.
                p_left_kitti = np.zeros((3, 4))
                # Cameras are always rectified.
                p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']

                # Create KITTI style transforms.
                velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
                velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

                # Check that the rotation has the same format as in KITTI.
                assert (velo_to_cam_trans[1:3] < 0).all()

                # Retrieve the token from the lidar.
                # Note that this may be confusing as the filename of the camera will
                # include the timestamp of the lidar,
                # not the camera.
                filename_cam_full = sd_record_cam['filename']
                filename_lid_full = sd_record_lid['filename']
                # token = '%06d' % token_idx # Alternative to use KITTI names.
                token_idx += 1

                # Convert image (jpg to png).
                src_im_path = os.path.join(
                    self.nusc.dataroot, filename_cam_full)
                dst_im_path = os.path.join(
                    image_folder, token_to_write + '.png')
                if not os.path.exists(dst_im_path):
                    im = Image.open(src_im_path)
                    im.save(dst_im_path, "PNG")

                # Convert lidar.
                # Note that we are only using a single sweep, instead of the commonly used n sweeps.
                src_lid_path = os.path.join(
                    self.nusc.dataroot, filename_lid_full)
                dst_lid_path = os.path.join(
                    lidar_folder, token_to_write + '.bin')

                pcl = LidarPointCloud.from_file(src_lid_path)
                # In KITTI lidar frame.
                pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)
                with open(dst_lid_path, "w") as lid_file:
                    pcl.points.T.tofile(lid_file)

                # Add to tokens.
                # tokens.append(token_to_write)

                # Create calibration file.
                kitti_transforms = dict()
                kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
                kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
                # Cameras are already rectified.
                kitti_transforms['R0_rect'] = r0_rect.rotation_matrix
                kitti_transforms['Tr_velo_to_cam'] = np.hstack(
                    (velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
                kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
                calib_path = os.path.join(
                    calib_folder, token_to_write + '.txt')
                with open(calib_path, "w") as calib_file:
                    for (key, val) in kitti_transforms.items():
                        val = val.flatten()
                        val_str = '%.12e' % val[0]
                        for v in val[1:]:
                            val_str += ' %.12e' % v
                        calib_file.write('%s: %s\n' % (key, val_str))

                # Write label file.
                label_path = os.path.join(
                    label_folder, token_to_write + '.txt')
                if os.path.exists(label_path):
                    print('Skipping existing file: %s' % label_path)
                    continue
                # else:
                #     print('Writing file: %s' % label_path)
                with open(label_path, "w") as label_file:
                    for sample_annotation_token in sample_annotation_tokens:
                        sample_annotation = self.nusc.get(
                            'sample_annotation', sample_annotation_token)

                        # Get box in LIDAR frame.
                        _, box_lidar_nusc, _ = self.nusc.get_sample_data(
                            lidar_token,
                            box_vis_level=BoxVisibility.NONE,
                            selected_anntokens=[sample_annotation_token]
                        )
                        box_lidar_nusc = box_lidar_nusc[0]

                        # Truncated: Set all objects to 0 which means untruncated.
                        truncated = 0.0

                        # Occluded: Set all objects to full visibility as this information is
                        # not available in nuScenes.
                        occluded = 0

                        # Convert nuScenes category to nuScenes detection challenge category.
                        detection_name = sample_annotation['category_name']
                        # category_to_detection_name(sample_annotation['category_name'])
                        # Skip categories that are not part of the nuScenes detection challenge.
                        if detection_name is None:
                            continue

                        # Convert from nuScenes to KITTI box format.
                        box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                            box_lidar_nusc,
                            Quaternion(matrix=velo_to_cam_rot),
                            velo_to_cam_trans, r0_rect
                        )

                        # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                        bbox_2d = KittiDB.project_kitti_box_to_image(
                            box_cam_kitti, p_left_kitti, imsize=imsize)
                        if bbox_2d is None and not self.get_all_detections:
                            continue
                        elif bbox_2d is None and self.get_all_detections:
                            bbox_2d = (-1.0, -1.0, -1.0, -1.0) # default KITTI bbox


                        # Set dummy score so we can use this file as result.
                        box_cam_kitti.score = 0

                        # Convert box to output string format.
                        output = KittiDB.box_to_string(
                            name=detection_name,
                            box=box_cam_kitti,
                            bbox_2d=bbox_2d,
                            truncation=truncated,
                            occlusion=occluded
                        )

                        # Write to disk.
                        label_file.write(output + '\n')

    def render_kitti(self, render_2d: bool = False) -> None:
        """
        Renders the annotations in the KITTI dataset from a lidar and a camera view.
        :param render_2d: Whether to render 2d boxes (only works for camera data).
        """
        if render_2d:
            print('Rendering 2d boxes from KITTI format')
        else:
            print('Rendering 3d boxes projected from 3d KITTI format')

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=[''])

        def get_transforms(token: str, root: str) -> dict:
            calib_filename = KittiDB.get_filepath(token, 'calib', root=root)

            lines = [line.rstrip() for line in open(calib_filename)]
            velo_to_cam = np.array(
                lines[5].strip().split(' ')[1:],
                dtype=np.float64
            )
            velo_to_cam.resize((3, 4))

            r0_rect = np.array(
                lines[4].strip().split(' ')[1:],
                dtype=np.float64
            )
            r0_rect.resize((3, 3))
            p_left = np.array(
                lines[2].strip().split(' ')[1:],
                dtype=np.float64
            )
            p_left.resize((3, 4))

            # Merge rectification and projection into one matrix.
            p_combined = np.eye(4)
            p_combined[:3, :3] = r0_rect
            p_combined = np.dot(p_left, p_combined)
            return {
                'velo_to_cam': {
                    'R': velo_to_cam[:, :3],
                    'T': velo_to_cam[:, 3]
                },
                'r0_rect': r0_rect,
                'p_left': p_left,
                'p_combined': p_combined,
            }

        kitti.get_transforms = get_transforms  # monkeypatching np.float32 -> 64
        # Create output folder.
        render_dir = os.path.join(self.nusc_kitti_dir, 'render')
        if not os.path.isdir(render_dir):
            os.mkdir(render_dir)

        # Render each image.
        tokens = kitti.tokens
        if self.samples_count is not None:
            tokens = tokens[:self.samples_count]

        for token in tqdm(tokens):
            # print('Rendering token dets to disk: %s' % token)
            for sensor in ['lidar', 'camera']:
                out_path = os.path.join(
                    render_dir, '%s_%s.png' % (token, sensor))
                kitti.render_sample_data(
                    token, sensor_modality=sensor, out_path=out_path, render_2d=render_2d)
                # Close the windows to avoid a warning of too many open windows.
                plt.close()

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples


if __name__ == '__main__':
    fire.Fire(KittiConverter)
