import cv2
import numpy as np
import pandas as pd
from collections import OrderedDict
import math

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_objects_with_history()

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            object_ids = list(self.objects.keys())
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if row >= len(object_ids) or col >= len(input_centroids):
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    if row < len(object_ids):  # Bounds check
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    if col < len(input_centroids):  # Bounds check
                        self.register(input_centroids[col])

        return self.get_objects_with_history()

    def get_objects_with_history(self):
        return self.objects.copy()

class SpermMotilityAnalyzer:
    def __init__(self):
        # Optimized tracker settings for 640x480 @ 49fps
        self.tracker = CentroidTracker(max_disappeared=15, max_distance=40)
        self.position_history = {}
        self.frame_data = []
        
        # Initial thresholds - will be updated adaptively
        self.fast_threshold = 8   # pixels per frame (fast erratic movement)
        self.slow_threshold = 2   # pixels per frame (slow linear movement)
        self.static_threshold = 0.5  # pixels per frame (accounting for minor detection jitter)
        
        # For adaptive threshold calculation
        self.velocity_samples = []
        self.adaptive_thresholds_set = False
        self.sample_frames = 100  # Number of frames to sample for adaptive thresholds
        
    def preprocess_frame(self, frame):
        """Refined preprocessing for precise sperm head detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # More conservative threshold to avoid over-detection
        # Use manual threshold instead of OTSU for better control
        _, bright_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Smaller top-hat kernel for more precise bright object detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Higher threshold for top-hat to be more selective
        _, tophat_thresh = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)
        
        # Combine both methods
        combined = cv2.bitwise_or(bright_thresh, tophat_thresh)
        
        # More aggressive cleaning to remove noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_clean)
        
        # Remove very small noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_ERODE, kernel_clean, iterations=1)
        
        # Very light dilation to slightly expand sperm heads only
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.dilate(cleaned, kernel_dilate, iterations=1)
        
        return result
    
    def filter_sperm_contours(self, contours):
        """Enhanced filtering to accept sperm with bright halos/glows"""
        valid_rects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Slightly relaxed area constraints to include halo patterns
            if area < 10 or area > 300:  # Allow slightly larger for halos
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter very small bounding boxes but allow larger for halos
            if w < 3 or h < 3 or w > 60 or h > 60:  
                continue
            
            # Calculate aspect ratio
            aspect_ratio = max(w, h) / min(w, h)
            
            # Check circularity for sperm heads
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Calculate solidity (filled area vs convex hull)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Calculate extent (contour area vs bounding rectangle area)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # RELAXED CRITERIA for different sperm patterns:
            
            # Pattern 1: Compact circular/oval sperm heads
            if circularity > 0.4 and solidity > 0.6:
                valid_rects.append((x, y, x + w, y + h))
                continue
            
            # Pattern 2: Elongated sperm (head + visible tail)
            if (circularity > 0.2 and aspect_ratio > 2.0 and aspect_ratio < 6.0 
                and solidity > 0.5):
                valid_rects.append((x, y, x + w, y + h))
                continue
            
            # Pattern 3: Sperm with bright halos (lower circularity, irregular shape)
            # These look like "flowers" or "dots with braces"
            if (circularity > 0.1 and circularity < 0.4 and  # Irregular but not too chaotic
                extent > 0.3 and  # Reasonable fill of bounding box
                solidity > 0.3 and  # Not too fragmented
                area > 15 and area < 250):  # Reasonable size for halo pattern
                valid_rects.append((x, y, x + w, y + h))
                continue
            
            # Pattern 4: Very bright, slightly bloomed sperm heads
            if (area > 20 and area < 200 and  # Medium size
                extent > 0.4 and  # Good fill of bounding box
                aspect_ratio < 3.0):  # Not too elongated
                valid_rects.append((x, y, x + w, y + h))
                continue
            
        return valid_rects
    
    def calculate_velocity(self, object_id, current_pos):
        """Calculate velocity and classify movement with adaptive thresholds"""
        if object_id not in self.position_history:
            self.position_history[object_id] = [current_pos]
            return 0, "new"
        
        # Keep last few positions for smoothing
        self.position_history[object_id].append(current_pos)
        if len(self.position_history[object_id]) > 5:
            self.position_history[object_id].pop(0)
        
        # Calculate average velocity over last few frames
        positions = self.position_history[object_id]
        if len(positions) < 2:
            return 0, "static"
        
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocity = math.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
        
        avg_velocity = sum(velocities) / len(velocities)
        
        # Collect velocity samples for adaptive threshold calculation
        if not self.adaptive_thresholds_set:
            self.velocity_samples.append(avg_velocity)
            
            # After collecting enough samples, calculate adaptive thresholds
            if len(self.velocity_samples) >= self.sample_frames:
                self.calculate_adaptive_thresholds()
        
        # Classify movement using current thresholds
        if avg_velocity > self.fast_threshold:
            return avg_velocity, "fast"
        elif avg_velocity > self.slow_threshold:
            return avg_velocity, "slow"
        elif avg_velocity > self.static_threshold:
            return avg_velocity, "static"
        else:
            return avg_velocity, "static"
    
    def calculate_adaptive_thresholds(self):
        """Calculate adaptive thresholds based on observed velocity distribution"""
        if len(self.velocity_samples) == 0:
            return
        
        velocities = np.array(self.velocity_samples)
        
        # Remove outliers (velocities > 95th percentile might be noise)
        p95 = np.percentile(velocities, 95)
        clean_velocities = velocities[velocities <= p95]
        
        if len(clean_velocities) == 0:
            return
        
        # Calculate statistics
        min_vel = np.min(clean_velocities)
        max_vel = np.max(clean_velocities)
        median_vel = np.median(clean_velocities)
        
        print(f"\nAdaptive Threshold Calculation:")
        print(f"Velocity range: {min_vel:.2f} - {max_vel:.2f} pixels/frame")
        print(f"Median velocity: {median_vel:.2f} pixels/frame")
        
        # Set adaptive thresholds
        # Static: bottom 30% of velocities
        # Slow: 30% - 70% of velocities  
        # Fast: top 30% of velocities
        self.static_threshold = np.percentile(clean_velocities, 30)
        self.slow_threshold = np.percentile(clean_velocities, 70)
        self.fast_threshold = max_vel  # Anything above 70th percentile is fast
        
        print(f"New thresholds:")
        print(f"Static <= {self.static_threshold:.2f} pixels/frame")
        print(f"Slow: {self.static_threshold:.2f} - {self.slow_threshold:.2f} pixels/frame") 
        print(f"Fast > {self.slow_threshold:.2f} pixels/frame")
        
        self.adaptive_thresholds_set = True
        print("Adaptive thresholds applied!\n")
        
    def debug_preprocessing(self, frame, frame_number=0):
        """Enhanced debug function to show detection reasoning"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = self.preprocess_frame(frame)
        
        # Find contours for debugging
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create debug image
        debug_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Draw all contours in red
        cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 1)
        
        # Analyze each contour and categorize
        pattern_counts = {"circular": 0, "elongated": 0, "halo": 0, "bloomed": 0, "rejected": 0}
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10 or area > 300:
                pattern_counts["rejected"] += 1
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3 or w > 60 or h > 60:
                pattern_counts["rejected"] += 1
                continue
            
            aspect_ratio = max(w, h) / min(w, h)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # Determine pattern type and color
            color = (128, 128, 128)  # Default gray for rejected
            pattern = "rejected"
            
            if circularity > 0.4 and solidity > 0.6:
                color = (0, 255, 0)  # Green for circular
                pattern = "circular"
                pattern_counts["circular"] += 1
            elif (circularity > 0.2 and aspect_ratio > 2.0 and aspect_ratio < 6.0 
                  and solidity > 0.5):
                color = (255, 255, 0)  # Cyan for elongated
                pattern = "elongated"
                pattern_counts["elongated"] += 1
            elif (circularity > 0.1 and circularity < 0.4 and extent > 0.3 
                  and solidity > 0.3 and area > 15 and area < 250):
                color = (255, 0, 255)  # Magenta for halo
                pattern = "halo"
                pattern_counts["halo"] += 1
            elif (area > 20 and area < 200 and extent > 0.4 and aspect_ratio < 3.0):
                color = (0, 255, 255)  # Yellow for bloomed
                pattern = "bloomed"
                pattern_counts["bloomed"] += 1
            else:
                pattern_counts["rejected"] += 1
            
            # Draw rectangle with pattern-specific color
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            
            # Add small text label
            cv2.putText(debug_img, pattern[:4], (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add comprehensive statistics
        total_valid = sum([pattern_counts[k] for k in pattern_counts if k != "rejected"])
        y_pos = 30
        cv2.putText(debug_img, f"Total contours: {len(contours)}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(debug_img, f"Valid sperm: {total_valid}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += 20
        cv2.putText(debug_img, f"Circular: {pattern_counts['circular']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_pos += 15
        cv2.putText(debug_img, f"Elongated: {pattern_counts['elongated']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y_pos += 15
        cv2.putText(debug_img, f"Halo: {pattern_counts['halo']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        y_pos += 15
        cv2.putText(debug_img, f"Bloomed: {pattern_counts['bloomed']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y_pos += 15
        cv2.putText(debug_img, f"Rejected: {pattern_counts['rejected']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # Save debug images
        cv2.imwrite(f'debug_frame_{frame_number}_original.jpg', frame)
        cv2.imwrite(f'debug_frame_{frame_number}_processed.jpg', processed)
        cv2.imwrite(f'debug_frame_{frame_number}_contours.jpg', debug_img)
        
        print(f"Debug Frame {frame_number}:")
        print(f"  Total: {len(contours)}, Valid: {total_valid}")
        print(f"  Circular: {pattern_counts['circular']}, Elongated: {pattern_counts['elongated']}")
        print(f"  Halo: {pattern_counts['halo']}, Bloomed: {pattern_counts['bloomed']}")
        print(f"  Rejected: {pattern_counts['rejected']}")
        
        return total_valid

    def analyze_video(self, video_path, output_dir="output", debug_mode=False):
        """Main analysis function with dual video outputs"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writers for both output styles
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Video 1: Dots with trajectories
        dots_video = cv2.VideoWriter(f'{output_dir}/sperm_analysis_dots_trajectories.mp4', 
                                   fourcc, fps, (width, height))
        
        # Video 2: Bounding boxes with trajectories  
        bbox_video = cv2.VideoWriter(f'{output_dir}/sperm_analysis_bbox_trajectories.mp4', 
                                   fourcc, fps, (width, height))
        
        frame_number = 0
        trajectory_history = {}  # Store trajectory points for each sperm
        max_trajectory_length = 30  # Number of points to keep in trajectory
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        print("Generating two output videos:")
        print("1. Dots with trajectories")
        print("2. Bounding boxes with trajectories")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Debug mode: save first few frames for analysis
            if debug_mode and frame_number <= 3:
                self.debug_preprocessing(frame, frame_number)
            
            # Preprocess frame
            processed = self.preprocess_frame(frame)
            
            # Find contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter for sperm-like objects
            sperm_rects = self.filter_sperm_contours(contours)
            
            # Update tracker
            objects = self.tracker.update(sperm_rects)
            
            # Analyze movement and create both video versions
            fast_count = slow_count = static_count = 0
            
            # Create copies for both video styles
            dots_frame = frame.copy()
            bbox_frame = frame.copy()
            
            for object_id, centroid in objects.items():
                velocity, movement_type = self.calculate_velocity(object_id, centroid)
                
                # Count by movement type and set colors
                if movement_type == "fast":
                    fast_count += 1
                    color = (0, 255, 0)  # Green for fast
                    color_name = "Fast"
                elif movement_type == "slow":
                    slow_count += 1
                    color = (0, 255, 255)  # Yellow for slow
                    color_name = "Slow"
                else:
                    static_count += 1
                    color = (0, 0, 255)  # Red for immotile
                    color_name = "Immotile"
                
                # Update trajectory history
                if object_id not in trajectory_history:
                    trajectory_history[object_id] = []
                
                trajectory_history[object_id].append(centroid)
                
                # Keep only recent trajectory points
                if len(trajectory_history[object_id]) > max_trajectory_length:
                    trajectory_history[object_id].pop(0)
                
                # Draw trajectories on both frames
                if len(trajectory_history[object_id]) > 1:
                    pts = trajectory_history[object_id]
                    for i in range(1, len(pts)):
                        # Calculate line thickness based on age (newer = thicker)
                        thickness = max(1, int(3 * (i / len(pts))))
                        cv2.line(dots_frame, pts[i-1], pts[i], color, thickness)
                        cv2.line(bbox_frame, pts[i-1], pts[i], color, thickness)
                
                # DOTS VIDEO: Draw filled circles
                cv2.circle(dots_frame, centroid, 6, color, -1)
                cv2.circle(dots_frame, centroid, 6, (255, 255, 255), 1)  # White border
                cv2.putText(dots_frame, f"{object_id}", 
                           (centroid[0] - 8, centroid[1] - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # BBOX VIDEO: Draw bounding rectangles
                # Find the original rectangle for this centroid
                min_dist = float('inf')
                best_rect = None
                for rect in sperm_rects:
                    rect_center = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
                    dist = np.sqrt((centroid[0] - rect_center[0])**2 + (centroid[1] - rect_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_rect = rect
                
                if best_rect and min_dist < 20:  # Only if close enough
                    cv2.rectangle(bbox_frame, (best_rect[0], best_rect[1]), 
                                (best_rect[2], best_rect[3]), color, 2)
                    cv2.putText(bbox_frame, f"{object_id}", 
                               (best_rect[0], best_rect[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add comprehensive statistics and color legend to both frames
            total_count = len(objects)
            
            # Statistics overlay (top-left) - with white text and black outline for visibility
            stats_y = 25
            stats_lines = [
                f"Frame: {frame_number}/{total_frames}",
                f"Total Sperm: {total_count}",
                f"Fast (Green): {fast_count}",
                f"Slow (Yellow): {slow_count}", 
                f"Immotile (Red): {static_count}"
            ]
            
            for i, line in enumerate(stats_lines):
                y_pos = stats_y + (i * 20)
                # White text with black outline for visibility on any background
                cv2.putText(dots_frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
                cv2.putText(dots_frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
                cv2.putText(bbox_frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
                cv2.putText(bbox_frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
            
            # Color legend (bottom-right) - with white text and black outline
            legend_x = width - 150
            legend_y = height - 70
            
            legend_items = [
                ("Fast Moving", (0, 255, 0)),
                ("Slow Moving", (0, 255, 255)), 
                ("Immotile", (0, 0, 255))
            ]
            
            for i, (label, color) in enumerate(legend_items):
                y_pos = legend_y + (i * 20)
                # Draw color indicator
                cv2.circle(dots_frame, (legend_x, y_pos), 8, color, -1)
                cv2.circle(bbox_frame, (legend_x, y_pos), 8, color, -1)
                # Draw label with outline for visibility
                cv2.putText(dots_frame, label, (legend_x + 20, y_pos + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Black outline
                cv2.putText(dots_frame, label, (legend_x + 20, y_pos + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # White text
                cv2.putText(bbox_frame, label, (legend_x + 20, y_pos + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Black outline
                cv2.putText(bbox_frame, label, (legend_x + 20, y_pos + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # White text
            
            # Save frame data with updated column names
            self.frame_data.append({
                'Frame #': frame_number,
                'Fast': fast_count,
                'Slow': slow_count,
                'Immotile': static_count
            })
            
            # Write both annotated frames
            dots_video.write(dots_frame)
            bbox_video.write(bbox_frame)
            
            # Progress indicator
            if frame_number % 50 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Cleanup
        cap.release()
        dots_video.release()
        bbox_video.release()
        
        # Save CSV report with new column names
        df = pd.DataFrame(self.frame_data)
        csv_path = f'{output_dir}/sperm_motility_analysis.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\nAnalysis complete!")
        print(f"CSV report: {csv_path}")
        print(f"Dots + trajectories video: {output_dir}/sperm_analysis_dots_trajectories.mp4")
        print(f"Bounding boxes + trajectories video: {output_dir}/sperm_analysis_bbox_trajectories.mp4")
        
        # Summary statistics with correct column names
        print(f"\nSummary:")
        print(f"Frames processed: {len(self.frame_data)}")
        total_sperm_avg = df['Fast'].mean() + df['Slow'].mean() + df['Immotile'].mean()
        print(f"Average sperm per frame: {total_sperm_avg:.1f}")
        print(f"Average fast-moving: {df['Fast'].mean():.1f}")
        print(f"Average slow-moving: {df['Slow'].mean():.1f}")
        print(f"Average immotile: {df['Immotile'].mean():.1f}")
        
        return df


if __name__ == "__main__":
    analyzer = SpermMotilityAnalyzer()
    
    video_path = "videos/12.mp4"
    
    print("Starting sperm motility analysis...")
    print("This may take a few minutes depending on video length...")
    
    # Run complete analysis
    results_df = analyzer.analyze_video(video_path, debug_mode=False)
    
    print("\nAnalysis completed successfully!")
    print("Check the output folder for results.")
    
    # Movement analysis
    total_fast = results_df['Fast'].sum()
    total_slow = results_df['Slow'].sum() 
    total_static = results_df['Immotile'].sum()
    total_detections = total_fast + total_slow + total_static
    
    if total_detections > 0:
        motile_percentage = (total_fast + total_slow) / total_detections * 100
        print(f"Overall Motility: {motile_percentage:.1f}%")
        
        if motile_percentage >= 40:
            classification = "Normal motility"
        elif motile_percentage >= 32:
            classification = "Below normal"  
        else:
            classification = "Poor motility"
        print(f"Classification: {classification}")
    
    print("Files generated in output/ folder:")
    print("- sperm_motility_analysis.csv")
    print("- sperm_analysis_dots_trajectories.mp4")
    print("- sperm_analysis_bbox_trajectories.mp4")