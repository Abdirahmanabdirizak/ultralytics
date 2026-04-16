# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""TrackTrack: Multi-cue tracker with HMIoU, iterative assignment, and track-aware initialization.

Based on "Focusing on Tracks for Online Multi-Object Tracking" (CVPR 2025).
Key features:
    - HMIoU (Height-aware Modified IoU) for better vertical overlap handling
    - Multi-cue cost: IoU + cosine ReID + confidence similarity + corner angle distance
    - Iterative assignment with progressively decreasing threshold
    - Track-Aware Initialization (TAI) NMS to suppress duplicate new tracks
    - Corner-based velocity model for angle distance computation
    - Confidence-aware Kalman filter with noise scaled by (1 - confidence)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


def _bbox_overlaps(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes in xyxy format.

    Args:
        a_xyxy (np.ndarray): Array of shape (N, 4) in [x1, y1, x2, y2] format.
        b_xyxy (np.ndarray): Array of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        (np.ndarray): IoU matrix of shape (N, M).
    """
    if len(a_xyxy) == 0 or len(b_xyxy) == 0:
        return np.zeros((len(a_xyxy), len(b_xyxy)), dtype=np.float64)

    inter_x1 = np.maximum(a_xyxy[:, 0:1], b_xyxy[:, 0:1].T)
    inter_y1 = np.maximum(a_xyxy[:, 1:2], b_xyxy[:, 1:2].T)
    inter_x2 = np.minimum(a_xyxy[:, 2:3], b_xyxy[:, 2:3].T)
    inter_y2 = np.minimum(a_xyxy[:, 3:4], b_xyxy[:, 3:4].T)

    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = (a_xyxy[:, 2] - a_xyxy[:, 0]) * (a_xyxy[:, 3] - a_xyxy[:, 1])
    area_b = (b_xyxy[:, 2] - b_xyxy[:, 0]) * (b_xyxy[:, 3] - b_xyxy[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-9)


def _hmiou_distance(a_tracks: list, b_tracks: list) -> tuple[np.ndarray, np.ndarray]:
    """Compute HMIoU (Height-aware Modified IoU) distance between tracks.

    HMIoU = HIoU * IoU, where HIoU is the ratio of vertical overlap to total vertical extent.
    Returns both the IoU similarity and HMIoU distance.

    Args:
        a_tracks (list): List of track objects with xyxy property.
        b_tracks (list): List of track objects with xyxy property.

    Returns:
        iou_sim (np.ndarray): IoU similarity matrix of shape (N, M).
        hmiou_dist (np.ndarray): HMIoU distance matrix of shape (N, M).
    """
    if len(a_tracks) == 0 or len(b_tracks) == 0:
        n, m = len(a_tracks), len(b_tracks)
        return np.zeros((n, m), dtype=np.float64), np.ones((n, m), dtype=np.float64)

    a_boxes = np.ascontiguousarray([t.xyxy for t in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([t.xyxy for t in b_tracks], dtype=np.float64)

    # Standard IoU
    iou_sim = _bbox_overlaps(a_boxes, b_boxes)

    # Height IoU (HIoU): vertical overlap / vertical union
    h_overlap = np.minimum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.maximum(a_boxes[:, 1:2], b_boxes[:, 1:2].T)
    h_union = np.maximum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.minimum(a_boxes[:, 1:2], b_boxes[:, 1:2].T)
    h_iou = np.clip(h_overlap / (h_union + 1e-9), 0, 1)

    # HMIoU = HIoU * IoU
    hmiou_sim = h_iou * iou_sim
    hmiou_dist = 1.0 - hmiou_sim

    return iou_sim, hmiou_dist


def _corner_velocity(box_prev: np.ndarray, box_curr: np.ndarray) -> np.ndarray:
    """Compute normalized velocity vectors for 4 corners (LT, LB, RT, RB).

    Both boxes are in xyxy format [x1, y1, x2, y2].

    Args:
        box_prev (np.ndarray): Previous box in xyxy format.
        box_curr (np.ndarray): Current box in xyxy format.

    Returns:
        (np.ndarray): Corner velocities of shape (4, 2).
    """
    deltas = box_curr - box_prev
    # Corner pairs: LT=(x1,y1), LB=(x1,y2), RT=(x2,y1), RB=(x2,y2)
    corners = [
        (deltas[0], deltas[1]),  # LT
        (deltas[0], deltas[3]),  # LB
        (deltas[2], deltas[1]),  # RT
        (deltas[2], deltas[3]),  # RB
    ]
    vel = np.zeros((4, 2), dtype=np.float64)
    for i, (dx, dy) in enumerate(corners):
        norm = np.sqrt(dx**2 + dy**2) + 1e-5
        vel[i] = [dx / norm, dy / norm]
    return vel


def _angle_distance(tracks: list, dets: list, frame_id: int, delta_t: int = 3) -> np.ndarray:
    """Compute angle-based distance between track velocities and track-to-detection directions.

    Vectorized: computes the full (N, M, 4, 2) velocity tensor in one pass.

    Args:
        tracks (list[TTSTrack]): Tracked objects with velocity and history.
        dets (list[TTSTrack]): Detection objects.
        frame_id (int): Current frame ID.
        delta_t (int): Time delta for velocity computation.

    Returns:
        (np.ndarray): Angle distance matrix of shape (N, M), values in [0, 1].
    """
    n_tracks, n_dets = len(tracks), len(dets)
    if n_tracks == 0 or n_dets == 0:
        return np.ones((n_tracks, n_dets), dtype=np.float64)

    # Stack boxes once
    track_boxes_prev = np.stack([t.get_history_box(frame_id, delta_t) for t in tracks], axis=0)  # (N, 4)
    det_boxes = np.stack([d.xyxy for d in dets], axis=0)  # (M, 4)

    # Delta per (track, det) pair: (N, M, 4)
    deltas = det_boxes[None, :, :] - track_boxes_prev[:, None, :]

    # Corner deltas: LT=(dx1,dy1), LB=(dx1,dy2), RT=(dx2,dy1), RB=(dx2,dy2)
    # Shape: (N, M, 4, 2) where last dim is (dx, dy)
    dx_idx = np.array([0, 0, 2, 2])
    dy_idx = np.array([1, 3, 1, 3])
    vel_dx = deltas[:, :, dx_idx]  # (N, M, 4)
    vel_dy = deltas[:, :, dy_idx]  # (N, M, 4)
    norms = np.sqrt(vel_dx * vel_dx + vel_dy * vel_dy) + 1e-5
    vel_dx /= norms
    vel_dy /= norms

    # Stack track velocities: (N, 4, 2)
    track_velocities = np.stack([t.velocity for t in tracks], axis=0)

    # Dot product per corner: (N, M, 4)
    dot = track_velocities[:, None, :, 0] * vel_dx + track_velocities[:, None, :, 1] * vel_dy
    angle_dist = np.abs(np.arccos(np.clip(dot, -1, 1))).mean(axis=-1) / np.pi  # (N, M)

    # Fuse with detection scores
    scores = np.array([d.score for d in dets])[None, :]
    angle_dist *= scores
    return angle_dist


def _confidence_distance(tracks: list, dets: list) -> np.ndarray:
    """Compute confidence-based distance using linear projection of track scores.

    Args:
        tracks (list[TTSTrack]): Tracked objects with score history.
        dets (list[TTSTrack]): Detection objects with scores.

    Returns:
        (np.ndarray): Confidence distance matrix of shape (N, M).
    """
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Get previous scores (one frame back)
    t_score_prev = np.array([t.prev_score for t in tracks])
    t_score = np.array([t.score for t in tracks])
    # Linear projection
    t_score_proj = t_score + (t_score - t_score_prev)

    d_score = np.array([d.score for d in dets])
    return np.abs(t_score_proj[:, None] - d_score[None, :])


def _iterative_associate(cost: np.ndarray, match_thr: float, reduce_step: float = 0.05):
    """Iteratively associate tracks and detections with decreasing threshold.

    Args:
        cost (np.ndarray): Cost matrix of shape (N, M).
        match_thr (float): Initial matching threshold.
        reduce_step (float): Amount to reduce threshold each iteration.

    Returns:
        matches (list[list[int]]): Matched (track_idx, det_idx) pairs.
        u_tracks (list[int]): Unmatched track indices.
        u_dets (list[int]): Unmatched detection indices.
    """
    matches = []
    cost = cost.copy()

    while True:
        # Find greedy minimum-cost matches
        new_matches = []
        if cost.shape[0] > 0 and cost.shape[1] > 0:
            min_det_per_track = np.argmin(cost, axis=1)
            min_track_per_det = np.argmin(cost, axis=0)
            for t_idx, d_idx in enumerate(min_det_per_track):
                if min_track_per_det[d_idx] == t_idx and cost[t_idx, d_idx] < match_thr:
                    new_matches.append([t_idx, d_idx])

        if len(new_matches) == 0:
            break

        matches.extend(new_matches)
        for t, d in new_matches:
            cost[t, :] = 1.0
            cost[:, d] = 1.0

        match_thr -= reduce_step

    m_tracks = {t for t, _ in matches}
    m_dets = {d for _, d in matches}
    u_tracks = [t for t in range(cost.shape[0]) if t not in m_tracks]
    u_dets = [d for d in range(cost.shape[1]) if d not in m_dets]

    return matches, u_tracks, u_dets


def _track_aware_nms(tracks: list, dets: list, tai_thr: float, init_thr: float) -> list[bool]:
    """Track-Aware Initialization NMS to suppress detections that overlap with existing tracks.

    Args:
        tracks (list): Existing active tracks.
        dets (list): New detection candidates.
        tai_thr (float): IoU threshold for suppression against tracks and other detections.
        init_thr (float): Minimum score for new track initialization.

    Returns:
        (list[bool]): Flags indicating which detections are allowed for initialization.
    """
    if len(dets) == 0:
        return []

    scores = np.array([d.score for d in dets])
    allow = list(scores > init_thr)

    # Compute pairwise IoU among all tracks + dets
    all_objs = tracks + dets
    if len(all_objs) < 2:
        return allow

    all_boxes = np.ascontiguousarray([o.xyxy for o in all_objs], dtype=np.float64)
    pair_iou = _bbox_overlaps(all_boxes, all_boxes)
    n_tracks = len(tracks)

    for i in range(len(dets)):
        if not allow[i]:
            continue
        # Check overlap with existing tracks
        if n_tracks > 0 and np.max(pair_iou[n_tracks + i, :n_tracks]) > tai_thr:
            allow[i] = False
            continue
        # Check overlap with higher-scoring detections
        for j in range(len(dets)):
            if i != j and allow[j] and scores[i] > scores[j]:
                if pair_iou[n_tracks + i, n_tracks + j] > tai_thr:
                    allow[j] = False

    return allow


class TTSTrack(BaseTrack):
    """Single object track for TrackTrack with corner velocity and ReID features.

    Attributes:
        shared_kalman (KalmanFilterXYWH): Shared Kalman filter for all TTSTrack instances.
        velocity (np.ndarray): Corner velocity vectors of shape (4, 2).
        feat (np.ndarray | None): ReID feature vector (EMA smoothed).
        curr_feat (np.ndarray | None): Current frame's raw feature.
        _history (dict): Frame-indexed history storing (box, score, mean, covariance).
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, xywh: list[float], score: float, cls: Any, feat: np.ndarray | None = None):
        """Initialize a TTSTrack instance.

        Args:
            xywh (list[float]): Bounding box in (x, y, w, h, idx) or (x, y, w, h, angle, idx) format.
            score (float): Confidence score.
            cls (Any): Class label.
            feat (np.ndarray | None): Optional ReID feature vector.
        """
        super().__init__()
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.prev_score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

        # Corner velocity (4 corners x 2D)
        self.velocity = np.zeros((4, 2), dtype=np.float64)
        self.delta_t = 3

        # History: frame_id -> (box_xyxy, score, mean, covariance)
        self._history: dict[int, tuple] = {}

        # ReID features
        self.smooth_feat = None
        self.curr_feat = None
        self._alpha = 0.95
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray):
        """Update feature vector with EMA smoothing.

        Args:
            feat (np.ndarray): New feature vector.
        """
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            beta = self._alpha + (1 - self._alpha) * (1 - self.score)
            self.smooth_feat = beta * self.smooth_feat + (1 - beta) * feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-9

    def get_history_box(self, frame_id: int, dt: int) -> np.ndarray:
        """Get box from history dt frames back, falling back to most recent.

        Args:
            frame_id (int): Current frame ID.
            dt (int): Frames to look back.

        Returns:
            (np.ndarray): Box in xyxy format.
        """
        target = frame_id - dt
        if target in self._history:
            return self._history[target][0].copy()
        if self._history:
            return self._history[max(self._history.keys())][0].copy()
        return self.xyxy.copy()

    def predict(self):
        """Predict the next state using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_gmc(stracks: list[TTSTrack], H: np.ndarray = np.eye(2, 3)):
        """Update multiple track positions and covariances using a homography matrix.

        Args:
            stracks (list[TTSTrack]): List of tracks to update.
            H (np.ndarray): 2x3 affine warp matrix.
        """
        if stracks:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_predict(stracks: list[TTSTrack]):
        """Perform batch Kalman filter prediction for multiple tracks."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = TTSTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYWH, frame_id: int):
        """Activate a new tracklet.

        Args:
            kalman_filter (KalmanFilterXYWH): Kalman filter instance.
            frame_id (int): Current frame ID.
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._history[frame_id] = (self.xyxy.copy(), self.score, self.mean.copy(), self.covariance.copy())

        self.tracklet_len = 0
        self.state = TrackState.New
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: TTSTrack, frame_id: int, new_id: bool = False):
        """Reactivate a previously lost track.

        Args:
            new_track (TTSTrack): New detection to reactivate with.
            frame_id (int): Current frame ID.
            new_id (bool): Whether to assign a new track ID.
        """
        self.prev_score = self.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self._history[frame_id] = (self.xyxy.copy(), new_track.score, self.mean.copy(), self.covariance.copy())
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track: TTSTrack, frame_id: int):
        """Update the track with a new matched detection.

        Args:
            new_track (TTSTrack): Matched detection.
            frame_id (int): Current frame ID.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.prev_score = self.score

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self._history[frame_id] = (new_track.xyxy.copy(), new_track.score, self.mean.copy(), self.covariance.copy())

        # Update corner velocity
        self.velocity = np.zeros((4, 2), dtype=np.float64)
        for dt in range(1, self.delta_t + 1):
            prev_box = self.get_history_box(frame_id, dt)
            self.velocity += _corner_velocity(prev_box, new_track.xyxy) / dt
        self.velocity /= self.delta_t

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert tlwh to xywh format for the Kalman filter."""
        return self.tlwh_to_xywh(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """Get bounding box in top-left-width-height format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Convert bounding box from tlwh to xyxy format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh to xywh (center-x, center-y, width, height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """Get position in (center x, center y, width, height, angle) format."""
        if self.angle is None:
            from ..utils import LOGGER

            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Get the current tracking results."""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"TT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class TRACKTRACK:
    """TrackTrack: Multi-cue tracker with iterative assignment and track-aware initialization.

    Implements the algorithm from "Focusing on Tracks for Online Multi-Object Tracking" (CVPR 2025).

    Attributes:
        tracked_stracks (list[TTSTrack]): Successfully activated tracks.
        lost_stracks (list[TTSTrack]): Lost tracks.
        removed_stracks (list[TTSTrack]): Removed tracks.
        frame_id (int): Current frame ID.
        args (Namespace): Tracker configuration.
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize a TRACKTRACK instance.

        Args:
            args (Namespace): Tracker configuration arguments.
            frame_rate (int): Frame rate of the video.
        """
        self.tracked_stracks: list[TTSTrack] = []
        self.lost_stracks: list[TTSTrack] = []
        self.removed_stracks: list[TTSTrack] = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

        # Association parameters
        self.det_thr = getattr(args, "det_thr", 0.6)
        self.match_thr = getattr(args, "match_thresh", 0.7)
        self.penalty_p = getattr(args, "penalty_p", 0.2)
        self.penalty_q = getattr(args, "penalty_q", 0.4)
        self.reduce_step = getattr(args, "reduce_step", 0.05)
        self.min_track_len = getattr(args, "min_track_len", 3)
        self.iou_weight = getattr(args, "iou_weight", 0.5)
        self.reid_weight = getattr(args, "reid_weight", 0.5)
        self.conf_weight = getattr(args, "conf_weight", 0.1)
        self.angle_weight = getattr(args, "angle_weight", 0.05)

        # Initialization parameters
        self.tai_thr = getattr(args, "tai_thr", 0.55)
        self.init_thr = getattr(args, "init_thr", 0.7)

        # GMC (camera motion compensation)
        gmc_method = getattr(args, "gmc_method", "sparseOptFlow")
        gmc_downscale = getattr(args, "gmc_downscale", 3)
        self.gmc = GMC(method=gmc_method, downscale=gmc_downscale)
        # Tune sparseOptFlow for speed: fewer corners suffice for affine motion estimation
        if gmc_method == "sparseOptFlow":
            gmc_max_corners = getattr(args, "gmc_max_corners", 200)
            self.gmc.feature_params["maxCorners"] = gmc_max_corners
        self._gmc_skip = getattr(args, "gmc_skip_frames", 0)
        self._gmc_warp_cached = np.eye(2, 3, dtype=np.float64)
        self._gmc_counter = 0

        # ReID
        self.with_reid = getattr(args, "with_reid", False)
        self.encoder = None
        if self.with_reid:
            model = getattr(args, "model", "auto")
            if model == "auto":
                self.encoder = lambda feats, s: [f.cpu().numpy() for f in feats]
            else:
                from .bot_sort import ReID

                self.encoder = ReID(model)

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None, dets_del=None) -> np.ndarray:
        """Update tracker with new detections.

        Args:
            results: Detection results with conf, xywh/xywhr, and cls attributes.
            img (np.ndarray | None): Current frame image (for ReID).
            feats (np.ndarray | None): Pre-extracted features from backbone.
            dets_del (tuple | None): Deleted detections (xywh, conf, cls) from loose NMS.

        Returns:
            (np.ndarray): Array of tracked objects [x1, y1, x2, y2, id, score, cls, idx].
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)

        # Split detections into high/low confidence
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)

        bboxes_keep = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        cls_keep = results.cls[remain_inds]

        bboxes_second = bboxes[inds_low]
        scores_second = scores[inds_low]
        cls_second = results.cls[inds_low]

        # Initialize detection objects
        if self.with_reid and self.encoder is not None and img is not None:
            features = self.encoder(img, bboxes_keep)  # ReID encoder crops from full image
            dets_high = [TTSTrack(b, s, c, f) for b, s, c, f in zip(bboxes_keep, scores_keep, cls_keep, features)]
        else:
            dets_high = [TTSTrack(b, s, c) for b, s, c in zip(bboxes_keep, scores_keep, cls_keep)]
        dets_low = [TTSTrack(b, s, c) for b, s, c in zip(bboxes_second, scores_second, cls_second)]

        # D_del: deleted high-confidence detections recovered from loose NMS (paper Eq. 1)
        dets_del_high = []
        if dets_del is not None:
            del_xywh, del_conf, del_cls = dets_del
            # Add dummy index column to match bboxes format
            del_bboxes = np.concatenate([del_xywh, -np.ones((len(del_xywh), 1))], axis=-1)
            # Only keep high-confidence deleted detections
            del_high_mask = del_conf > self.det_thr
            if del_high_mask.any():
                dets_del_high = [TTSTrack(b, s, c) for b, s, c in
                                 zip(del_bboxes[del_high_mask], del_conf[del_high_mask], del_cls[del_high_mask])]

        # Split existing tracks
        tracked_stracks = []
        unconfirmed = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Pool tracked + lost tracks
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # Camera motion compensation (optionally skip frames for speed)
        if img is not None:
            if self._gmc_skip > 0 and self._gmc_counter % (self._gmc_skip + 1) != 0:
                warp = self._gmc_warp_cached
            else:
                warp = self.gmc.apply(img, [t.xyxy for t in dets_high])
                self._gmc_warp_cached = warp
            self._gmc_counter += 1
            TTSTrack.multi_gmc(strack_pool, warp)
            TTSTrack.multi_gmc(unconfirmed, warp)

        # Predict
        self.multi_predict(strack_pool)

        # === Main association: tracked+lost vs high+low+del detections (paper Eq. 1) ===
        all_dets = dets_high + dets_low + dets_del_high
        n_high = len(dets_high)
        n_low = len(dets_low)

        # Compute multi-cue cost matrix
        iou_sim, hmiou_dist = _hmiou_distance(strack_pool, all_dets)
        cost = self.iou_weight * hmiou_dist

        # Add cosine distance if ReID is enabled
        if self.with_reid and self.encoder is not None:
            cos_dist = self._cosine_distance(strack_pool, all_dets)
            cost += self.reid_weight * cos_dist
        else:
            cost += self.reid_weight * hmiou_dist  # Fall back to HMIoU for the ReID weight

        # Add confidence and angle distances
        cost += self.conf_weight * _confidence_distance(strack_pool, all_dets)
        cost += self.angle_weight * _angle_distance(strack_pool, all_dets, self.frame_id)

        # Penalize low-confidence and deleted detections (paper Eq. 1)
        if cost.shape[1] > n_high:
            cost[:, n_high:n_high + n_low] += self.penalty_p  # τ_p for D_low
        if dets_del_high and cost.shape[1] > n_high + n_low:
            cost[:, n_high + n_low:] += self.penalty_q  # τ_q for D_del

        # Constrain: if IoU is too low, set cost to 1
        if iou_sim.size > 0:
            cost[iou_sim <= 0.10] = 1.0
        cost = np.clip(cost, 0, 1)

        # Iterative assignment
        matches, u_track, u_det = _iterative_associate(cost, self.match_thr, self.reduce_step)

        for t_idx, d_idx in matches:
            track = strack_pool[t_idx]
            det = all_dets[d_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Mark unmatched tracks as lost
        for t_idx in u_track:
            track = strack_pool[t_idx]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # === Association: unconfirmed tracks vs remaining high-confidence detections ===
        remaining_high_dets = [all_dets[i] for i in u_det if i < n_high]
        if unconfirmed and remaining_high_dets:
            uc_iou_sim, uc_hmiou_dist = _hmiou_distance(unconfirmed, remaining_high_dets)
            uc_cost = self.iou_weight * uc_hmiou_dist
            if self.with_reid and self.encoder is not None:
                uc_cost += self.reid_weight * self._cosine_distance(unconfirmed, remaining_high_dets)
            else:
                uc_cost += self.reid_weight * uc_hmiou_dist
            uc_cost += self.conf_weight * _confidence_distance(unconfirmed, remaining_high_dets)
            uc_cost += self.angle_weight * _angle_distance(unconfirmed, remaining_high_dets, self.frame_id)
            if uc_iou_sim.size > 0:
                uc_cost[uc_iou_sim <= 0.10] = 1.0
            uc_cost = np.clip(uc_cost, 0, 1)
            uc_matches, uc_u_track, uc_u_det = _iterative_associate(uc_cost, self.match_thr, self.reduce_step)
            for t_idx, d_idx in uc_matches:
                unconfirmed[t_idx].update(remaining_high_dets[d_idx], self.frame_id)
                activated_stracks.append(unconfirmed[t_idx])
            for t_idx in uc_u_track:
                unconfirmed[t_idx].mark_removed()
                removed_stracks.append(unconfirmed[t_idx])
            remaining_high_dets = [remaining_high_dets[i] for i in uc_u_det]
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed_stracks.append(track)

        # === Track-Aware Initialization ===
        active_tracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        active_tracks.extend(activated_stracks)
        allow_flags = _track_aware_nms(active_tracks, remaining_high_dets, self.tai_thr, self.init_thr)
        for i, det in enumerate(remaining_high_dets):
            if i < len(allow_flags) and allow_flags[i]:
                det.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(det)

        # === Cleanup ===
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return np.asarray(
            [x.result for x in self.tracked_stracks if x.is_activated and x.frame_id == self.frame_id],
            dtype=np.float32,
        )

    @staticmethod
    def _cosine_distance(tracks: list[TTSTrack], dets: list[TTSTrack]) -> np.ndarray:
        """Compute cosine distance between track and detection features.

        Args:
            tracks (list[TTSTrack]): Tracks with smooth_feat.
            dets (list[TTSTrack]): Detections with curr_feat.

        Returns:
            (np.ndarray): Cosine distance matrix of shape (N, M).
        """
        if len(tracks) == 0 or len(dets) == 0:
            return np.ones((len(tracks), len(dets)), dtype=np.float64)

        # Determine feature dimension from first available feature
        dim = 128
        for obj in (*tracks, *dets):
            f = obj.smooth_feat if obj.smooth_feat is not None else obj.curr_feat
            if f is not None:
                dim = f.shape[0]
                break

        t_feat = np.stack([t.smooth_feat if t.smooth_feat is not None else np.zeros(dim, dtype=np.float32) for t in tracks])
        d_feat = np.stack([d.curr_feat if d.curr_feat is not None else np.zeros(dim, dtype=np.float32) for d in dets])
        return np.clip(1 - np.dot(t_feat, d_feat.T), 0, 1)

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        """Return a KalmanFilterXYWH instance."""
        return KalmanFilterXYWH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[TTSTrack]:
        """Initialize tracks from detection results."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [TTSTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def multi_predict(self, tracks: list[TTSTrack]):
        """Predict the next states for multiple tracks."""
        TTSTrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset the ID counter."""
        TTSTrack.reset_id()

    def reset(self):
        """Reset the tracker state."""
        self.tracked_stracks: list[TTSTrack] = []
        self.lost_stracks: list[TTSTrack] = []
        self.removed_stracks: list[TTSTrack] = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
        self.gmc.reset_params()

    @staticmethod
    def joint_stracks(tlista: list[TTSTrack], tlistb: list[TTSTrack]) -> list[TTSTrack]:
        """Combine two lists of tracks, ensuring no duplicates by track_id."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            if not exists.get(t.track_id, 0):
                exists[t.track_id] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[TTSTrack], tlistb: list[TTSTrack]) -> list[TTSTrack]:
        """Remove tracks in tlistb from tlista."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(
        stracksa: list[TTSTrack], stracksb: list[TTSTrack]
    ) -> tuple[list[TTSTrack], list[TTSTrack]]:
        """Remove duplicate tracks based on IoU distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
