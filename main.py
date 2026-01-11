import cv2
import numpy as np

class Solution:
    """
    Mandatory Solution class for the Vehant Vehicle Count Challenge.
    This class uses classical image processing techniques to estimate 
    the total number of vehicles moving away from a static camera.
    """

    def __init__(self):
        """
        Initialize the processing pipeline configuration.
        Classical CV thresholds and tracker state are defined here.
        """
        # Multi-Tiered Geometric Thresholds for different vehicle types
        self.min_area_car = 2000         # Minimum area threshold for cars
        self.min_area_bike = 700         # Lower threshold for motorcycle detection
        self.aspect_ratio_car = 1.2      # Height/Width ratio limit for cars
        self.aspect_ratio_bike = 2.1     # Height/Width ratio limit for bikes
        self.grad_threshold = 18         # Sobel gradient sensitivity for structural edges
        
        # Dynamic Gate State: Calculated automatically per video
        self.p1, self.p2 = (0, 0), (0, 0)
        self.is_night = False            # Ambient light state flag
        
        # Tracker State: Maintains unique vehicle identities across frames
        self.tracked_vehicles = []       
        self.max_lost_frames = 20        # Tolerance for temporary object occlusion
        self.dist_threshold = 100        # Pixel radius for temporal matching
        self.total_count = 0             # Resulting vehicle tally

    def _detect_environment(self, bg_gray):
        """Senses ambient light to toggle specific pre-processing modes."""
        self.is_night = np.mean(bg_gray) < 65

    def _initialize_dynamic_gate(self, bg_gray):
        """Automatically aligns a virtual tripwire with the detected road geometry."""
        h, w = bg_gray.shape
        edges = cv2.Canny(bg_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=h//4, maxLineGap=50)
        
        y_lefts, y_rights = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 20: continue 
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                y0 = int(intercept)
                yw = int(slope * w + intercept)
                # Keep points within the middle region of the frame
                if 0.1 * h < y0 < 0.9 * h: y_lefts.append(y0)
                if 0.1 * h < yw < 0.9 * h: y_rights.append(yw)
                
        # Fallback to perspective defaults if road edges aren't dominant
        self.p1 = (0, int(np.median(y_lefts)) if y_lefts else int(h * 0.75))
        self.p2 = (w, int(np.median(y_rights)) if y_rights else int(h * 0.45))

    def get_master_background(self, video_path):
        """Generates a median background reference image to isolate motion."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        frames = []
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample frames throughout the video for a robust median
        for i in range(0, total_f, max(1, total_f // 40)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret or frame is None: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        
        if not frames: return None
        median_bg = np.median(frames, axis=0).astype(np.uint8)
        self._detect_environment(median_bg)
        self._initialize_dynamic_gate(median_bg)
        return median_bg

    def _get_adaptive_mask(self, gray_f, bg_gray):
        """Creates a binary motion mask adapted for environmental lighting."""
        if self.is_night:
            # Enhanced night mode pre-processing
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            gray_f = clahe.apply(gray_f)
            bg_gray = clahe.apply(bg_gray)
            b_size, t_val, c_size = (13, 13), 25, (23, 23)
        else:
            # Standard daylight pre-processing
            b_size, t_val, c_size = (7, 7), 35, (15, 15)
            
        diff = cv2.absdiff(cv2.GaussianBlur(gray_f, b_size, 0), 
                           cv2.GaussianBlur(bg_gray, b_size, 0))
        _, mask = cv2.threshold(diff, t_val, 255, cv2.THRESH_BINARY)
        
        # Clean mask noise and fuse fragmented vehicle blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, c_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _get_threshold_y(self, x):
        """Calculates the specific Y-coordinate on the virtual tripwire for an X position."""
        return self.p1[1] + (self.p2[1] - self.p1[1]) * (x - self.p1[0]) / (self.p2[0] - self.p1[0])

    def _process_frame_logic(self, frame, bg_gray, frame_h):
        """Identifies vehicle detections in a frame using structural and geometric rules."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._get_adaptive_mask(gray, bg_gray)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = h / float(w)
            
            # Perspective Factor: Scale thresholds based on vertical depth in the frame
            p_factor = (y / frame_h)
            
            # Rule 1: Multi-tiered classification for cars and bikes
            is_car = (area > self.min_area_car * p_factor and aspect_ratio < self.aspect_ratio_car)
            is_bike = (area > self.min_area_bike * p_factor and aspect_ratio < self.aspect_ratio_bike)
            
            if not (is_car or is_bike): continue
            
            # Rule 2: Rigid Human Exclusion (verticality limit)
            if aspect_ratio > 2.2: continue 
            
            # Rule 3: Texture verification using horizontal gradient analysis
            roi = gray[y:y+h, x:x+w]
            if np.mean(np.abs(cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3))) > self.grad_threshold:
                detections.append((x + w // 2, y + h // 2))
                
        return detections

    def _update_tracker(self, detections):
        """Associates frame detections with tracked vehicles and tallies new counts."""
        new_tracked = []
        for v in self.tracked_vehicles:
            # Estimate next position based on previous frame velocity
            pred_x = v['centroid'][0] + v.get('vx', 0)
            pred_y = v['centroid'][1] + v.get('vy', 0)
            
            best_dist, best_idx = float('inf'), -1
            for i, pos in enumerate(detections):
                dist = np.hypot(pos[0] - pred_x, pos[1] - pred_y)
                if dist < best_dist and dist < self.dist_threshold:
                    best_dist, best_idx = dist, i
            
            if best_idx != -1:
                new_pos = detections.pop(best_idx)
                # Update velocity and position
                v['vx'] = new_pos[0] - v['centroid'][0]
                v['vy'] = new_pos[1] - v['centroid'][1]
                v['centroid'] = new_pos
                v['lost'] = 0
                
                # Increment count if the ID crosses the virtual tripwire for the first time
                tripwire_y = self._get_threshold_y(v['centroid'][0])
                if not v['counted'] and v['centroid'][1] < tripwire_y:
                    self.total_count += 1
                    v['counted'] = True
                new_tracked.append(v)
            else:
                # Keep tracking 'lost' objects briefly to handle occlusion
                v['lost'] += 1
                if v['lost'] < self.max_lost_frames:
                    new_tracked.append(v)
        
        # Initiate tracking for new detections appearing on the entry side of the gate
        for pos in detections:
            if pos[1] > self._get_threshold_y(pos[0]):
                new_tracked.append({'centroid': pos, 'counted': False, 'lost': 0, 'vx': 0, 'vy': 0})
        self.tracked_vehicles = new_tracked

    def forward(self, video_path: str) -> int:
        """
        Main entry point for vehicle counting as per mandatory hackathon guidelines.
        
        Args:
            video_path (str): The absolute file path to the input traffic video.
            
        Returns:
            int: The total calculated vehicle count.
        """
        # Step 1: Generate reference background and calibrate geometry
        bg_gray = self.get_master_background(video_path)
        if bg_gray is None: return 0
        
        cap = cv2.VideoCapture(video_path)
        self.total_count = 0
        self.tracked_vehicles = []
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Step 2: Sequential Frame Processing
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: 
                break
            
            # Identify detections and update tracker
            detections = self._process_frame_logic(frame, bg_gray, frame_h)
            self._update_tracker(detections)
            
        cap.release()
        return int(self.total_count)