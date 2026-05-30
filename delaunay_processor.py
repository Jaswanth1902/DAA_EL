import cv2
import numpy as np
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans

class DelaunayProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image at {image_path}")
        self.h, self.w = self.image.shape[:2]
        self.points = []
        self.triangles = [] # List of (pts, color)

    def _get_seeds(self, k=15, b_dist=5, interior_density=0.001):
        """
        Implementation of the technical sampling pipeline:
        1. K-Means segmentation (optimized with adaptive downsampling)
        2. Noise reduction and boundary extraction
        3. Boundary-aware seed point sampling (b_dist)
        4. Random interior insertion
        """
        # 1. Image segmentation via k-means clustering into k color-similar regions
        # Downsample for K-Means if the image is large to speed up processing and smooth out high-frequency noise
        max_dim = 512
        if max(self.h, self.w) > max_dim:
            scale = max_dim / max(self.h, self.w)
            small_img = cv2.resize(self.image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small_img = self.image

        pixels = small_img.reshape(-1, 3).astype(np.float32)
        
        # Use MiniBatchKMeans if available for ultra-fast clustering
        try:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=3, random_state=42)
        except ImportError:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            
        labels = kmeans.fit_predict(pixels)
        
        if max(self.h, self.w) > max_dim:
            small_segmented = labels.reshape(small_img.shape[0], small_img.shape[1]).astype(np.uint8)
            segmented = cv2.resize(small_segmented, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        else:
            segmented = labels.reshape(self.h, self.w).astype(np.uint8)

        # Apply a median blur to reduce pixel fragmentation and noise before edge detection
        segmented = cv2.medianBlur(segmented, 5)

        # 2. Extract region boundaries
        # Use gradient or Canny on the segmented labels to find region boundaries
        edges = cv2.Canny(segmented, 0, 1) # Low thresholds because labels are distinct
        
        # 3. Boundary-aware seed point sampling (spaced by Bpointdist)
        y_idx, x_idx = np.where(edges > 0)
        boundary_points = np.column_stack((x_idx, y_idx))
        
        if len(boundary_points) > 0:
            indices = np.arange(0, len(boundary_points), b_dist)
            self.points.extend(boundary_points[indices].tolist())

        # 4. Interior random point insertion for equilateral triangles/coverage
        num_interior = int(self.w * self.h * interior_density)
        interior_x = np.random.randint(0, self.w, num_interior)
        interior_y = np.random.randint(0, self.h, num_interior)
        self.points.extend(np.column_stack((interior_x, interior_y)).tolist())

        # Always include corners to ensure full image coverage
        self.points.extend([[0, 0], [0, self.h-1], [self.w-1, 0], [self.w-1, self.h-1]])

    def process(self, k=15, b_dist=5, interior_density=0.001):
        self.points = []
        self._get_seeds(k=k, b_dist=b_dist, interior_density=interior_density)

        # Remove duplicate points
        self.points = np.unique(self.points, axis=0)
        
        # 6. Compute Delaunay Triangulation
        tri = Delaunay(self.points)
        
        # Pre-allocate mask to avoid overhead of allocating memory for every triangle
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        self.triangles = []
        for simplex in tri.simplices:
            pts = self.points[simplex].astype(np.int32)
            
            # Clear previous mask
            mask.fill(0)
            cv2.drawContours(mask, [pts], 0, 255, -1)
            
            # Compute average color c
            mean_color = cv2.mean(self.image, mask=mask)[:3]
            
            # Fallback for extremely small/collinear triangles that cover 0 pixel centers
            if sum(mean_color) == 0:
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                cx = min(max(0, cx), self.w - 1)
                cy = min(max(0, cy), self.h - 1)
                mean_color = self.image[cy, cx].tolist()
                
            self.triangles.append((pts, mean_color))

    def render(self):
        """Render the triangulated version I."""
        output = np.zeros_like(self.image)
        for pts, color in self.triangles:
            cv2.drawContours(output, [pts], 0, color, -1)
        return output
