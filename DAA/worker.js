importScripts('https://unpkg.com/delaunator@5.0.0/delaunator.min.js');

self.onmessage = function(e) {
    const { action, pixelData, w, h, s, cachedCoords } = e.data;
    
    if (action === 'computeMesh') {
        const result = computeMesh(pixelData, w, h, s);
        self.postMessage({ action: 'computeMeshResult', ...result });
    } else if (action === 'recomputeTriangulation') {
        const result = recomputeTriangulation(cachedCoords, pixelData, w, h);
        self.postMessage({ action: 'recomputeTriangulationResult', ...result });
    }
};

// Helper to check if point p is inside triangle p0,p1,p2
function pointInTriangle(px, py, p0, p1, p2) {
    const dX = px - p2[0];
    const dY = py - p2[1];
    const dX21 = p2[0] - p1[0];
    const dY12 = p1[1] - p2[1];
    const D = dY12 * (p0[0] - p2[0]) + dX21 * (p0[1] - p2[1]);
    const s = dY12 * dX + dX21 * dY;
    const t = (p2[1] - p0[1]) * dX + (p0[0] - p2[0]) * dY;
    if (D < 0) return s <= 0 && t <= 0 && s + t >= D;
    return s >= 0 && t >= 0 && s + t <= D;
}

// Dominant Color Extraction (replaces the muddy average/centroid approach)
function getDominantColor(coords, indices, pixelData, w, h) {
    const colMap = [];
    
    for (let t = 0; t < indices.length; t += 3) {
        const p0 = coords[indices[t]];
        const p1 = coords[indices[t + 1]];
        const p2 = coords[indices[t + 2]];

        const minX = Math.max(0, Math.floor(Math.min(p0[0], p1[0], p2[0])));
        const maxX = Math.min(w - 1, Math.ceil(Math.max(p0[0], p1[0], p2[0])));
        const minY = Math.max(0, Math.floor(Math.min(p0[1], p1[1], p2[1])));
        const maxY = Math.min(h - 1, Math.ceil(Math.max(p0[1], p1[1], p2[1])));

        // We use a small sampling grid inside the bounding box for speed
        const step = Math.max(1, Math.floor(Math.min(maxX - minX, maxY - minY) / 5));
        
        let samples = [];
        for (let y = minY; y <= maxY; y += step) {
            for (let x = minX; x <= maxX; x += step) {
                if (pointInTriangle(x, y, p0, p1, p2)) {
                    const cIdx = (y * w + x) * 4;
                    samples.push({
                        r: pixelData[cIdx],
                        g: pixelData[cIdx + 1],
                        b: pixelData[cIdx + 2]
                    });
                }
            }
        }
        
        if (samples.length === 0) {
            // Fallback to centroid if triangle is extremely thin/small
            const cx = Math.floor((p0[0] + p1[0] + p2[0]) / 3);
            const cy = Math.floor((p0[1] + p1[1] + p2[1]) / 3);
            const cIdx = (cy * w + cx) * 4;
            colMap.push([pixelData[cIdx]||0, pixelData[cIdx + 1]||0, pixelData[cIdx + 2]||0]);
            continue;
        }

        // Quantize colors into bins to find the dominant color
        const bins = {};
        let maxBin = null;
        let maxCount = 0;
        
        for (let i = 0; i < samples.length; i++) {
            const s = samples[i];
            const q = 32; // Quantization step
            const binKey = `${Math.floor(s.r/q)},${Math.floor(s.g/q)},${Math.floor(s.b/q)}`;
            if (!bins[binKey]) bins[binKey] = { r:0, g:0, b:0, count:0 };
            bins[binKey].r += s.r;
            bins[binKey].g += s.g;
            bins[binKey].b += s.b;
            bins[binKey].count++;
            
            if (bins[binKey].count > maxCount) {
                maxCount = bins[binKey].count;
                maxBin = bins[binKey];
            }
        }
        
        // Return average of the dominant bin
        colMap.push([
            Math.round(maxBin.r / maxBin.count),
            Math.round(maxBin.g / maxBin.count),
            Math.round(maxBin.b / maxBin.count)
        ]);
    }
    
    return colMap;
}

function computeMesh(data, w, h, s) {
    const t0 = performance.now();
    
    // Step 2: Sobel Edge Filtering
    const edges = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            const g00 = (data[((y - 1) * w + (x - 1)) * 4] + data[((y - 1) * w + (x - 1)) * 4 + 1] + data[((y - 1) * w + (x - 1)) * 4 + 2]) / 3;
            const g01 = (data[((y - 1) * w + x) * 4] + data[((y - 1) * w + x) * 4 + 1] + data[((y - 1) * w + x) * 4 + 2]) / 3;
            const g02 = (data[((y - 1) * w + (x + 1)) * 4] + data[((y - 1) * w + (x + 1)) * 4 + 1] + data[((y - 1) * w + (x + 1)) * 4 + 2]) / 3;
            const g10 = (data[(y * w + (x - 1)) * 4] + data[(y * w + (x - 1)) * 4 + 1] + data[(y * w + (x - 1)) * 4 + 2]) / 3;
            const g12 = (data[(y * w + (x + 1)) * 4] + data[(y * w + (x + 1)) * 4 + 1] + data[(y * w + (x + 1)) * 4 + 2]) / 3;
            const g20 = (data[((y + 1) * w + (x - 1)) * 4] + data[((y + 1) * w + (x - 1)) * 4 + 1] + data[((y + 1) * w + (x - 1)) * 4 + 2]) / 3;
            const g21 = (data[((y + 1) * w + x) * 4] + data[((y + 1) * w + x) * 4 + 1] + data[((y + 1) * w + x) * 4 + 2]) / 3;
            const g22 = (data[((y + 1) * w + (x + 1)) * 4] + data[((y + 1) * w + (x + 1)) * 4 + 1] + data[((y + 1) * w + (x + 1)) * 4 + 2]) / 3;

            const gx = (g02 + 2 * g12 + g22) - (g00 + 2 * g10 + g20);
            const gy = (g20 + 2 * g21 + g22) - (g00 + 2 * g01 + g02);

            const mag = Math.sqrt(gx * gx + gy * gy);
            edges[y * w + x] = mag > s.threshold ? 255 : 0;
        }
    }

    // Step 3: Sample point grid
    let coords = [];
    const spacing = s.spacing;
    
    // Sample edge points spaced out
    for (let y = 1; y < h - 1; y += 2) {
        for (let x = 1; x < w - 1; x += 2) {
            if (edges[y * w + x] === 255) {
                let tooClose = false;
                for (let k = Math.max(0, coords.length - 100); k < coords.length; k++) {
                    const dx = coords[k][0] - x;
                    const dy = coords[k][1] - y;
                    if (dx * dx + dy * dy < spacing * spacing) {
                        tooClose = true;
                        break;
                    }
                }
                if (!tooClose) {
                    coords.push([x, y]);
                }
            }
        }
    }

    // Sample random interior points
    const interiorPointsCount = Math.floor(w * h * s.density);
    for (let i = 0; i < interiorPointsCount; i++) {
        coords.push([Math.floor(Math.random() * w), Math.floor(Math.random() * h)]);
    }

    // Anchor corners
    coords.push([0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]);

    // Remove duplicates
    const seen = new Set();
    coords = coords.filter(p => {
        const key = `${p[0]},${p[1]}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });

    const flatPoints = new Float64Array(coords.length * 2);
    for (let i = 0; i < coords.length; i++) {
        flatPoints[i * 2] = coords[i][0];
        flatPoints[i * 2 + 1] = coords[i][1];
    }

    const d = new Delaunator(flatPoints);
    const indices = Array.from(d.triangles); // convert to standard array

    const colMap = getDominantColor(coords, indices, data, w, h);
    
    const t1 = performance.now();
    return { coords, indices, colMap, time: t1 - t0, edges: Array.from(edges) };
}

function recomputeTriangulation(coords, data, w, h) {
    if (coords.length < 3) return { indices: [], colMap: [] };

    const flatPoints = new Float64Array(coords.length * 2);
    for (let i = 0; i < coords.length; i++) {
        flatPoints[i * 2] = coords[i][0];
        flatPoints[i * 2 + 1] = coords[i][1];
    }

    const d = new Delaunator(flatPoints);
    const indices = Array.from(d.triangles);

    const colMap = getDominantColor(coords, indices, data, w, h);

    return { indices, colMap };
}
