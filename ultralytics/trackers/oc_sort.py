# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching


class OCSortTrack(STrack):
    """Track object for OC-SORT with observation-centric state management.

    Extends STrack with storage of real detector observations and velocity computation,
    enabling the three OC-SORT components: ORU, OCM, and OCR.

    Attributes:
        last_observation (np.ndarray): Last real detection in xyxy format.
        observations (dict): Maps frame_id to xyxy observation arrays.
        velocity (np.ndarray | None): Observation-centric velocity direction vector (dx, dy).
        delta_t (int): Temporal window for velocity computation.
    """

    def __init__(self, xywh: list[float], score: float, cls: Any, delta_t: int = 3):
        """Initialize OCSortTrack with observation storage.

        Args:
            xywh (list[float]): Bounding box in (x, y, w, h, idx) or (x, y, w, h, angle, idx) format.
            score (float): Confidence score.
            cls (Any): Class label.
            delta_t (int): Temporal window for velocity direction computation.
        """
        super().__init__(xywh, score, cls)
        self.last_observation = np.array([-1, -1, -1, -1], dtype=np.float32)
        self.observations: dict[int, np.ndarray] = {}
        self.velocity = None
        self.delta_t = delta_t
        self._saved_mean = None
        self._saved_covariance = None

    def activate(self, kalman_filter, frame_id: int):
        """Activate a new tracklet and record the initial observation."""
        super().activate(kalman_filter, frame_id)
        self.last_observation = self.xyxy.copy()
        self.observations[frame_id] = self.xyxy.copy()
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()

    def update(self, new_track: STrack, frame_id: int):
        """Update track state with a new matched detection and record the observation."""
        obs = new_track.xyxy.copy()
        self.last_observation = obs
        self.observations[frame_id] = obs
        self._prune_observations(frame_id)
        super().update(new_track, frame_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()
        self.velocity = self._compute_velocity()

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Reactivate a lost track with a new detection and record the observation."""
        obs = new_track.xyxy.copy()
        self.last_observation = obs
        self.observations[frame_id] = obs
        super().re_activate(new_track, frame_id, new_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()
        self.velocity = self._compute_velocity()

    @staticmethod
    def _xyxy_center(xyxy):
        """Return (cx, cy) center of an xyxy bounding box."""
        return np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])

    def _prune_observations(self, current_frame: int):
        """Remove old observations to prevent unbounded memory growth. Keeps recent entries for velocity."""
        max_keep = self.delta_t + 2
        if len(self.observations) <= max_keep:
            return
        sorted_frames = sorted(self.observations.keys())
        for frame in sorted_frames[:-max_keep]:
            del self.observations[frame]

    def _compute_velocity(self):
        """Compute observation-centric velocity direction from stored observations.

        Returns:
            (np.ndarray | None): Normalized direction vector (dx, dy) or None if insufficient observations.
        """
        if len(self.observations) < 2:
            return None

        current_frame = max(self.observations.keys())
        current_center = self._xyxy_center(self.observations[current_frame])

        # Find the most recent observation at least delta_t frames before current
        prev_obs = None
        for frame in sorted(self.observations.keys(), reverse=True):
            if frame < current_frame - self.delta_t + 1:
                prev_obs = self.observations[frame]
                break

        # Fallback: use the earliest observation if nothing is delta_t frames back
        if prev_obs is None:
            earliest_frame = min(self.observations.keys())
            if earliest_frame == current_frame:
                return None
            prev_obs = self.observations[earliest_frame]

        direction = current_center - self._xyxy_center(prev_obs)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (direction / norm).astype(np.float32)

    def apply_oru(self, new_observation_xyxy: np.ndarray, current_frame_id: int):
        """Apply Observation-Centric Re-Update: interpolate virtual observations and replay Kalman updates.

        This repairs the Kalman state after occlusion by retroactively updating with linearly
        interpolated virtual observations between the last real observation and the new detection.

        Args:
            new_observation_xyxy (np.ndarray): New detection in xyxy format.
            current_frame_id (int): Current frame ID.
        """
        if self._saved_mean is None or not self.observations:
            return

        last_frame = max(self.observations.keys())
        gap = current_frame_id - last_frame
        if gap <= 1:
            return

        # Restore Kalman state to last observation point
        self.mean = self._saved_mean.copy()
        self.covariance = self._saved_covariance.copy()

        last_obs = self.observations[last_frame]

        # Replay with virtual observations
        for t in range(1, gap):
            alpha = t / gap
            virtual_xyxy = (1 - alpha) * last_obs + alpha * new_observation_xyxy
            # Convert xyxy to tlwh then to xyah for Kalman measurement
            virtual_tlwh = np.array([
                virtual_xyxy[0],
                virtual_xyxy[1],
                virtual_xyxy[2] - virtual_xyxy[0],
                virtual_xyxy[3] - virtual_xyxy[1],
            ])
            virtual_xyah = self.tlwh_to_xyah(virtual_tlwh)
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, virtual_xyah)

        # Final predict to reach current frame
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)


class OCSORT(BYTETracker):
    """OC-SORT multi-object tracker with observation-centric association.

    Implements three key components on top of BYTETracker:
    - Observation-Centric Re-Update (ORU): repairs Kalman state after occlusion
    - Observation-Centric Momentum (OCM): velocity direction consistency cost
    - Observation-Centric Recovery (OCR): re-association using last observation position

    Attributes:
        delta_t (int): Temporal window for velocity direction computation.
        inertia (float): Weight of velocity consistency cost in association.
        use_byte (bool): Whether to use ByteTrack-style low-confidence second pass.
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize OC-SORT tracker.

        Args:
            args (Namespace): Tracking parameters including delta_t, inertia, use_byte.
            frame_rate (int): Video frame rate.
        """
        super().__init__(args, frame_rate)
        self.delta_t = args.delta_t
        self.inertia = args.inertia
        self.use_byte = args.use_byte

    def init_track(self, results, img=None):
        """Initialize OCSortTrack instances from detection results."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [OCSortTrack(xywh, s, c, self.delta_t) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks, detections):
        """Compute cost matrix with Buffered IoU distance and OCM velocity direction consistency cost."""
        dists = self._biou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        vel_dists = self._velocity_direction_cost(tracks, detections)
        dists = dists + self.inertia * vel_dists
        return dists

    @staticmethod
    def _biou_distance(tracks, detections, buffer_ratio=0.5):
        """Compute Buffered IoU distance: expand boxes before IoU to handle fast motion.

        Each box is expanded by buffer_ratio * sqrt(w*h) in all directions (C-BIoU, BoostTrack).

        Args:
            tracks: List of tracks or np.ndarray boxes.
            detections: List of detections or np.ndarray boxes.
            buffer_ratio (float): Expansion ratio relative to sqrt(area).

        Returns:
            (np.ndarray): Cost matrix (1 - BIoU).
        """
        def _expand_xyxy(xyxy, ratio):
            """Expand an xyxy box by ratio * sqrt(w*h) in all directions."""
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            buf = ratio * np.sqrt(max(w * h, 1.0))
            return np.array([xyxy[0] - buf, xyxy[1] - buf, xyxy[2] + buf, xyxy[3] + buf])

        def _get_boxes(items):
            if items and isinstance(items[0], np.ndarray):
                return items
            return [t.xywha if t.angle is not None else t.xyxy for t in items]

        aboxes = _get_boxes(tracks)
        bboxes = _get_boxes(detections)

        # Only expand xyxy (4-element) boxes, not rotated (5-element)
        aboxes_exp = [_expand_xyxy(b, buffer_ratio) if len(b) == 4 else b for b in aboxes]
        bboxes_exp = [_expand_xyxy(b, buffer_ratio) if len(b) == 4 else b for b in bboxes]

        return matching.iou_distance(aboxes_exp, bboxes_exp)

    def update(self, results, img=None, feats=None):
        """Update tracker with new detections using OC-SORT association pipeline."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results = results[remain_inds]

        detections = self.init_track(results, img)

        # Separate into confirmed and unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Stage 1: First association with high-score detections
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                # ORU: repair Kalman state before re-activation for lost tracks
                track.apply_oru(det.xyxy, self.frame_id)
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # OCR: Observation-Centric Recovery pass
        # Re-associate unmatched tracked (not lost) using last_observation position
        ocr_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        ocr_dets = [detections[i] for i in u_detection]

        if ocr_tracked and ocr_dets:
            ocr_dists = self._ocr_distance(ocr_tracked, ocr_dets)
            if self.args.fuse_score:
                ocr_dists = matching.fuse_score(ocr_dists, ocr_dets)
            ocr_matches, ocr_u_track, ocr_u_det = matching.linear_assignment(ocr_dists, thresh=self.args.match_thresh)

            for itracked, idet in ocr_matches:
                track = ocr_tracked[itracked]
                det = ocr_dets[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.apply_oru(det.xyxy, self.frame_id)
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            # Update unmatched sets after OCR
            ocr_u_track_set = {id(ocr_tracked[i]) for i in ocr_u_track}
            ocr_u_det_set = {id(ocr_dets[i]) for i in ocr_u_det}
            u_track = [i for i in u_track if id(strack_pool[i]) in ocr_u_track_set or strack_pool[i].state != TrackState.Tracked]
            u_detection = [i for i in u_detection if id(detections[i]) in ocr_u_det_set]

        # Stage 2: Low-confidence second pass (optional, ByteTrack-style)
        if self.use_byte:
            detections_second = self.init_track(results_second, img)
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, detections_second)
            matches, u_track_second, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            for it in u_track_second:
                track = r_tracked_stracks[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            # Mark unmatched tracked as lost (when not using byte second pass)
            for i in u_track:
                track = strack_pool[i]
                if track.state == TrackState.Tracked:
                    track.mark_lost()
                    lost_stracks.append(track)

        # Stage 3: Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Stage 4: Init new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Stage 5: Update state
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
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def _velocity_direction_cost(self, tracks, detections):
        """Compute OCM velocity direction consistency cost matrix (vectorized).

        For each track-detection pair, measures the angular difference between
        the track's historical motion direction and the direction to the candidate detection.

        Args:
            tracks (list[OCSortTrack]): List of tracks.
            detections (list[OCSortTrack]): List of detections.

        Returns:
            (np.ndarray): Cost matrix of shape (len(tracks), len(detections)).
        """
        n_tracks, n_dets = len(tracks), len(detections)
        cost = np.zeros((n_tracks, n_dets), dtype=np.float32)
        if cost.size == 0:
            return cost

        # Pre-extract detection centers as (M, 2) array
        det_centers = np.array([OCSortTrack._xyxy_center(det.xyxy) for det in detections], dtype=np.float32)

        for i, track in enumerate(tracks):
            if track.velocity is None or track.last_observation[0] < 0:
                continue
            track_center = OCSortTrack._xyxy_center(track.last_observation)
            directions = det_centers - track_center  # (M, 2)
            norms = np.linalg.norm(directions, axis=1)  # (M,)
            valid = norms > 1e-6
            if not valid.any():
                continue
            directions[valid] /= norms[valid, None]
            dots = np.clip(directions[valid] @ track.velocity, -1.0, 1.0)
            cost[i, valid] = np.arccos(dots) / np.pi

        return cost

    def _ocr_distance(self, tracks, detections):
        """Compute IoU distance using tracks' last observation positions instead of Kalman predictions.

        Args:
            tracks (list[OCSortTrack]): List of tracks with last_observation attributes.
            detections (list[OCSortTrack]): List of detections.

        Returns:
            (np.ndarray): Cost matrix based on IoU with last observations.
        """
        atlbrs = []
        for track in tracks:
            if track.last_observation[0] >= 0:
                obs = track.last_observation
            else:
                obs = track.xyxy
            atlbrs.append(obs if track.angle is None else track.xywha)
        btlbrs = [det.xywha if det.angle is not None else det.xyxy for det in detections]
        return matching.iou_distance(atlbrs, btlbrs)
