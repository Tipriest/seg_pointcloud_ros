#!/usr/bin/env python3
import struct

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2

from mmseg.apis import init_model, inference_model
from mmseg.visualization import SegLocalVisualizer


class SegPointCloudNode:
    def __init__(self):
        self.bridge = CvBridge()

        config = rospy.get_param("~config", "")
        checkpoint = rospy.get_param("~checkpoint", "")
        device = rospy.get_param("~device", "cuda:0")
        self.alpha = float(rospy.get_param("~alpha", 0.5))
        self.sample_rate = int(rospy.get_param("~sample_rate", 100))
        self.depth_scale = float(rospy.get_param("~depth_scale", 1000.0))
        self.frame_id = rospy.get_param("~frame_id", "")
        self.resize_width = int(rospy.get_param("~resize_width", 0))
        self.resize_height = int(rospy.get_param("~resize_height", 0))
        self.palette_param = rospy.get_param("~palette", [])

        intrinsics_param = rospy.get_param("~intrinsics", [])
        extrinsics_param = rospy.get_param("~extrinsics", [])

        if not config or not checkpoint:
            raise RuntimeError("~config and ~checkpoint params are required")
        if not intrinsics_param or not extrinsics_param:
            raise RuntimeError("~intrinsics and ~extrinsics params are required")

        self.fx, self.fy, self.cx, self.cy = self._parse_intrinsics(intrinsics_param)
        self.extrinsics = self._parse_extrinsics(extrinsics_param)

        self.model = init_model(config, checkpoint, device=device)
        self.visualizer = SegLocalVisualizer(alpha=self.alpha)
        self.visualizer.dataset_meta = self.model.dataset_meta
        if self.palette_param:
            self.visualizer.dataset_meta = dict(self.visualizer.dataset_meta or {})
            self.visualizer.dataset_meta["palette"] = self._parse_palette(
                self.palette_param
            )

        in_rgb = rospy.get_param("~input_rgb", "rgb")
        in_depth = rospy.get_param("~input_depth", "depth")
        out_cloud = rospy.get_param("~output_cloud", "seg_cloud")
        out_overlay = rospy.get_param("~output_overlay", "seg_overlay")

        self.sub_rgb = message_filters.Subscriber(in_rgb, Image)
        self.sub_depth = message_filters.Subscriber(in_depth, Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth], queue_size=5, slop=0.05
        )
        self.sync.registerCallback(self.cb)

        self.pub_cloud = rospy.Publisher(out_cloud, PointCloud2, queue_size=1)
        self.pub_overlay = rospy.Publisher(out_overlay, Image, queue_size=1)

    def _parse_intrinsics(self, intrinsics_param):
        if len(intrinsics_param) == 4:
            fx, fy, cx, cy = [float(v) for v in intrinsics_param]
            return fx, fy, cx, cy
        if len(intrinsics_param) == 9:
            k = [float(v) for v in intrinsics_param]
            fx, fy = k[0], k[4]
            cx, cy = k[2], k[5]
            return fx, fy, cx, cy
        raise RuntimeError("~intrinsics must have 4 or 9 values")

    def _parse_extrinsics(self, extrinsics_param):
        if len(extrinsics_param) != 16:
            raise RuntimeError("~extrinsics must have 16 values")
        mat = np.array([float(v) for v in extrinsics_param], dtype=np.float32)
        return mat.reshape(4, 4)

    def _get_palette(self, num_classes):
        if self.palette_param:
            return self._parse_palette(self.palette_param)

        dataset_meta = getattr(self.model, "dataset_meta", None)
        if dataset_meta and "palette" in dataset_meta:
            palette = dataset_meta["palette"]
            return np.array(palette, dtype=np.uint8)

        if num_classes <= 0:
            num_classes = 256
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        for i in range(num_classes):
            r = (37 * i) % 255
            g = (17 * i) % 255
            b = (29 * i) % 255
            palette[i] = [r, g, b]
        return palette

    def _parse_palette(self, palette_param):
        palette = np.array(palette_param, dtype=np.int32)
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise RuntimeError("~palette must be a list of [r, g, b] entries")
        palette = np.clip(palette, 0, 255).astype(np.uint8)
        return palette

    def _pack_rgb(self, r, g, b):
        rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]

    def cb(self, rgb_msg, depth_msg):
        rospy.loginfo("Received synchronized RGB and depth images")
        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        if self.resize_width > 0 and self.resize_height > 0:
            bgr = cv2.resize(bgr, (self.resize_width, self.resize_height))

        result = inference_model(self.model, bgr)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)

        depth_h, depth_w = depth.shape[:2]
        rgb_h, rgb_w = bgr.shape[:2]
        if (depth_h, depth_w) != (rgb_h, rgb_w):
            bgr = cv2.resize(bgr, (depth_w, depth_h))
            pred = cv2.resize(pred, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)

        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) / self.depth_scale
        else:
            depth_m = depth.astype(np.float32)

        valid_mask = np.isfinite(depth_m) & (depth_m > 0)

        u, v = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
        u = u[valid_mask]
        v = v[valid_mask]
        depth_vals = depth_m[valid_mask]
        labels = pred[valid_mask]

        if self.sample_rate > 1:
            u = u[:: self.sample_rate]
            v = v[:: self.sample_rate]
            depth_vals = depth_vals[:: self.sample_rate]
            labels = labels[:: self.sample_rate]

        x = (u - self.cx) * depth_vals / self.fx
        y = (v - self.cy) * depth_vals / self.fy
        z = depth_vals

        points_camera = np.vstack((x, y, z)).T
        ones = np.ones((points_camera.shape[0], 1), dtype=np.float32)
        points_h = np.hstack((points_camera, ones))
        points_world = (self.extrinsics @ points_h.T).T[:, :3]

        num_classes = int(np.max(labels)) + 1 if labels.size > 0 else 0
        palette = self._get_palette(num_classes)
        colors = palette[labels % palette.shape[0]]

        cloud_points = []
        for i in range(points_world.shape[0]):
            r, g, b = colors[i]
            rgb_float = self._pack_rgb(r, g, b)
            cloud_points.append(
                (points_world[i, 0], points_world[i, 1], points_world[i, 2], rgb_float)
            )

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]

        header = depth_msg.header
        if self.frame_id:
            header.frame_id = self.frame_id

        cloud_msg = point_cloud2.create_cloud(header, fields, cloud_points)
        self.pub_cloud.publish(cloud_msg)
        rospy.loginfo(f"Published segmented point cloud with {len(cloud_points)} points")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.visualizer.add_datasample(
            name="overlay",
            image=rgb,
            data_sample=result,
            draw_gt=False,
            draw_pred=True,
            show=False,
            out_file=None,
            with_labels=False,
        )
        overlay_rgb = self.visualizer.get_image()
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
        overlay_msg.header = rgb_msg.header
        self.pub_overlay.publish(overlay_msg)


if __name__ == "__main__":
    rospy.init_node("seg_pointcloud_node")
    node = SegPointCloudNode()
    rospy.spin()
