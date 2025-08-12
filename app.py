# app.py
"""
Real-Time Image Processing Studio (Streamlit)
Covers: Image acquisition, 1D/2D DFT, DCT, sampling/aliasing demo, histogram processing,
spatial filters, homomorphic filtering, morphological ops, skeletonization,
and block-DCT (JPEG-like) compression demo — without any external dataset or pre-trained model.

Run:
    streamlit run app.py

Requirements (pip):
    streamlit numpy opencv-python matplotlib scikit-image
"""

import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from io import BytesIO

# Optional: for skeletonization
try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

st.set_page_config(page_title="Image Processing Studio", layout="wide", initial_sidebar_state="expanded")
st.title("📷 Real-Time Image Acquisition & Processing Studio")
st.markdown("Upload an image or use the camera. Explore transforms, sampling, enhancement, morphology, and compression — all live.")

# -------------------------
# Helpers
# -------------------------
def to_rgb(img_bgr):
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def to_gray(img_rgb):
    if img_rgb is None:
        return None
    if img_rgb.ndim == 3:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return img_rgb

def pil_bytes_from_cv2(img_rgb):
    # convert RGB numpy to PNG bytes for download
    is_success, im_buf_arr = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return im_buf_arr.tobytes()

def show_image_col(img, caption="", width=None):
    if img is None:
        return
    if img.ndim == 2:
        st.image(img, caption=caption, width=width, clamp=True, channels="GRAY")
    else:
        st.image(img, caption=caption, width=width, clamp=True, channels="RGB")

# -------------------------
# Acquisition
# -------------------------
st.sidebar.header("Image acquisition")
source = st.sidebar.radio("Source", ["Upload Image", "Webcam (camera_input)"])
img_rgb = None

if source == "Upload Image":
    uploaded = st.sidebar.file_uploader("Choose image file", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = to_rgb(bgr)
    else:
        st.warning("Upload an image to start. (Or switch to Webcam.)")
elif source == "Webcam (camera_input)":
    cam = st.sidebar.camera_input("Use your camera")
    if cam is not None:
        # streamlit camera_input returns a UploadedFile-like object with bytes
        data = cam.getvalue()
        file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = to_rgb(bgr)
    else:
        st.info("Allow camera and take a photo or upload an image.")

if img_rgb is None:
    st.stop()

# Show original in left column
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Original / Acquisition")
    show_image_col(img_rgb, caption="Original (RGB)")

# -------------------------
# Sidebar: operation selection & common params
# -------------------------
st.sidebar.header("Processing modules")
module = st.sidebar.selectbox("Choose module", [
    "Transforms (DFT / DCT)",
    "Sampling & Aliasing demo",
    "Enhancement & Filters",
    "Morphology & Segmentation",
    "Compression (Block DCT / JPEG-like)",
    "Save / Download"
])

# -------------------------
# MODULE: Transforms
# -------------------------
def compute_dft(gray):
    # returns magnitude and phase scaled for display
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    phase = np.angle(fshift)
    return magnitude, phase, fshift

def apply_dct_block(img_gray):
    # Compute DCT of full image using cv2.dct (float32)
    h, w = img_gray.shape
    # pad to even dimensions if needed
    H = cv2.getOptimalDFTSize(h)
    W = cv2.getOptimalDFTSize(w)
    padded = np.zeros((H, W), dtype=np.float32)
    padded[:h, :w] = img_gray.astype(np.float32)
    dct = cv2.dct(padded)
    return dct[:h, :w]

def block_dct_compress(img_gray, block=8, quality=50):
    """Apply block-wise DCT (NxN), quantize coefficients to simulate JPEG compression.
       quality: 1 (worst) - 100 (best)"""
    img = img_gray.astype(np.float32)
    h, w = img.shape
    # pad to multiple of block
    ph = (block - (h % block)) % block
    pw = (block - (w % block)) % block
    padded = np.pad(img, ((0, ph), (0, pw)), mode='reflect')
    H, W = padded.shape
    # simple quant matrix scaled by quality
    Q50 = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99],
    ], dtype=np.float32)
    # scale Q50 by quality parameter
    if quality < 50:
        scale = 5000 / (quality * 100.0)
    else:
        scale = 2 - (quality * 2 / 100.0)
    Q = np.clip((Q50 * scale), 1, 255)
    rec = np.zeros_like(padded)
    for i in range(0, H, block):
        for j in range(0, W, block):
            block_pixels = padded[i:i+block, j:j+block]
            d = cv2.dct(block_pixels)
            # quantize
            dq = np.round(d / Q)
            # dequantize
            dd = dq * Q
            rb = cv2.idct(dd)
            rec[i:i+block, j:j+block] = rb
    rec = rec[:h, :w]
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    return rec

# -------------------------
# MODULE: Sampling demo
# -------------------------
def down_up_sample(img_rgb, factor=4, interpolation=cv2.INTER_NEAREST):
    h, w = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (w//factor, h//factor), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=interpolation)
    return up, small

# -------------------------
# MODULE: Homomorphic filter
# -------------------------
def homomorphic_filter(img_gray, low_gain=0.5, high_gain=1.5, cutoff=30):
    # Convert to float and log
    rows, cols = img_gray.shape
    img_log = np.log1p(img_gray.astype(np.float32))
    # Create Gaussian highpass filter in frequency domain
    M = 2*rows
    N = 2*cols
    y, x = np.ogrid[:M, :N]
    center = (M/2, N/2)
    dist = (x - center[1])**2 + (y - center[0])**2
    # butterworth-like gaussian
    H = 1 - np.exp(-dist/(2*(cutoff**2)))
    H = H[:M, :N]
    # Apply
    F = np.fft.fft2(img_log, (M, N))
    F_shift = np.fft.fftshift(F)
    G = (low_gain + high_gain*H) * F_shift
    G_ishift = np.fft.ifftshift(G)
    img_back = np.fft.ifft2(G_ishift)
    img_back = np.real(img_back)[:rows, :cols]
    img_exp = np.expm1(img_back)
    img_exp = np.clip(img_exp, 0, 255)
    return img_exp.astype(np.uint8)

# -------------------------
# MODULE: basic morphological skeleton (fallback)
# -------------------------
def naive_skeleton(img_bin):
    # Not very efficient; iterative thinning using morphological ops
    img = img_bin.copy()
    skel = np.zeros(img.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

# -------------------------
# MODULE: Wiener-like denoise (simple)
# -------------------------
def simple_wiener(img_gray, kernel_size=5):
    # A simple local variance-based filter (not full Wiener from skimage)
    # Use cv2.blur to compute local mean and variance
    img = img_gray.astype(np.float32)
    local_mean = cv2.blur(img, (kernel_size, kernel_size))
    local_mean_sq = cv2.blur(img**2, (kernel_size, kernel_size))
    local_var = local_mean_sq - local_mean**2
    noise_var = np.mean(local_var) * 0.1  # estimate
    result = local_mean + (np.maximum(local_var - noise_var, 0) / (local_var + 1e-8)) * (img - local_mean)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# -------------------------
# Main interactive modules
# -------------------------
with col2:
    st.subheader("Processing / Output")

    if module == "Transforms (DFT / DCT)":
        mode = st.selectbox("Transform", ["Show 2D DFT Magnitude & Phase", "2D DCT (full image)", "IDCT reconstruct (block)", "Show log-magnitude + phase"])
        gray = to_gray(img_rgb).astype(np.uint8)
        if mode.startswith("Show"):
            mag, phase, fshift = compute_dft(gray)
            # Normalize for display
            mag_disp = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            phase_disp = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.write("Magnitude Spectrum:")
            show_image_col(mag_disp, caption="DFT Magnitude (log scaled)")
            st.write("Phase Spectrum:")
            show_image_col(phase_disp, caption="DFT Phase (scaled)")
            # Show inverse reconstruct to verify
            if st.checkbox("Reconstruct inverse DFT and show error"):
                # inverse
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.real(img_back)
                img_back = np.clip(img_back, 0, 255).astype(np.uint8)
                show_image_col(img_back, caption="Inverse DFT Reconstruction")
                diff = cv2.absdiff(gray, img_back)
                show_image_col(diff, caption="Absolute difference (orig - recon)")
        elif mode == "2D DCT (full image)":
            dct = apply_dct_block(gray)
            # scale for viewing
            dct_disp = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            show_image_col(dct_disp, caption="2D DCT (float32 scaled)")
            if st.checkbox("Show IDCT of full-image DCT"):
                # pad back to same for idct
                H, W = dct.shape
                rec = cv2.idct(np.pad(dct, ((0, H-H),(0, W-W)), mode='constant') if False else dct)
                rec = np.clip(rec, 0, 255).astype(np.uint8)
                show_image_col(rec, caption="IDCT Reconstruction (approx)")
        elif mode == "IDCT reconstruct (block)":
            block = st.slider("Block size (for block IDCT sim)", 4, 16, 8)
            quality = st.slider("Quantization strength for experiment (lower -> heavier)", 1, 100, 50)
            rec = block_dct_compress(gray, block=block, quality=quality)
            st.write("Block-DCT compressed & reconstructed image (demonstrates transform coding):")
            show_image_col(rec, caption=f"Block DCT reconstructed (block={block}, quality={quality})")
            diff = cv2.absdiff(gray, rec)
            show_image_col(diff, caption="Absolute difference (orig - rec)")
        else:
            mag, phase, _ = compute_dft(gray)
            mag_disp = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            phase_disp = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            show_image_col(mag_disp, caption="DFT Magnitude (log scaled)")
            show_image_col(phase_disp, caption="DFT Phase (scaled)")

    elif module == "Sampling & Aliasing demo":
        st.write("Downsampling (simulate sampling) and upsampling (reconstruction). See aliasing vs interpolation.")
        factor = st.slider("Downsample factor", 2, 16, 4)
        interp = st.selectbox("Upsampling interpolation", ["Nearest (shows aliasing)", "Linear", "Cubic"])
        inter_map = {"Nearest": cv2.INTER_NEAREST, "Linear": cv2.INTER_LINEAR, "Cubic": cv2.INTER_CUBIC}
        up, small = down_up_sample(img_rgb, factor=factor, interpolation=inter_map[interp])
        st.write("Downsampled (small) image:")
        show_image_col(small, caption=f"Downsampled by {factor}")
        st.write("Upsampled back to original size using chosen interpolation:")
        show_image_col(up, caption=f"Upsampled using {interp}")
        st.write("Side-by-side comparison:")
        cols = st.columns(3)
        with cols[0]:
            show_image_col(img_rgb, caption="Original")
        with cols[1]:
            show_image_col(up, caption="Reconstructed")
        with cols[2]:
            diff = cv2.absdiff(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(up, cv2.COLOR_RGB2GRAY))
            show_image_col(diff, caption="Difference (gray absdiff)")

        st.markdown("**Sampling theorem demo idea:** Lowering sampling (higher downsample factor) will cause high-frequency content to alias into lower frequencies (visible as jagged / distorted textures). Using a proper anti-aliasing lowpass before decimation reduces aliasing.")

    elif module == "Enhancement & Filters":
        st.write("Spatial and frequency-domain enhancement techniques.")
        gray = to_gray(img_rgb).astype(np.uint8)
        choice = st.selectbox("Enhancement operation", [
            "Histogram Equalization (global)",
            "CLAHE (local equalization)",
            "Homomorphic filtering (frequency)",
            "Smoothing (Gaussian)",
            "Sharpening (unsharp mask)",
            "Wiener-like denoising"
        ])
        if choice == "Histogram Equalization (global)":
            if gray.ndim == 2:
                heq = cv2.equalizeHist(gray)
                show_image_col(heq, caption="Histogram Equalized (global)")
            else:
                show_image_col(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), caption="Input gray")
        elif choice == "CLAHE (local equalization)":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(gray)
            show_image_col(cl, caption="CLAHE result")
        elif choice == "Homomorphic filtering (frequency)":
            lg = st.slider("Low gain (illumination)", 0.1, 2.0, 0.5)
            hg = st.slider("High gain (reflectance)", 1.0, 5.0, 1.5)
            cutoff = st.slider("Cutoff (frequency)", 5, 80, 30)
            result = homomorphic_filter(gray, low_gain=lg, high_gain=hg, cutoff=cutoff)
            show_image_col(result, caption=f"Homomorphic filter (lg={lg}, hg={hg}, cutoff={cutoff})")
        elif choice == "Smoothing (Gaussian)":
            k = st.slider("Kernel size", 1, 31, 5, step=2)
            blurred = cv2.GaussianBlur(img_rgb, (k, k), 0)
            show_image_col(blurred, caption=f"Gaussian blur k={k}")
        elif choice == "Sharpening (unsharp mask)":
            k = st.slider("Blur kernel", 1, 31, 5, step=2)
            blurred = cv2.GaussianBlur(img_rgb, (k,k), 0)
            alpha = st.slider("Amount (alpha)", 0.1, 3.0, 1.0)
            sharp = cv2.addWeighted(img_rgb, 1+alpha, blurred, -alpha, 0)
            show_image_col(sharp, caption=f"Unsharp mask alpha={alpha}, blur_k={k}")
        elif choice == "Wiener-like denoising":
            ks = st.slider("Local kernel size", 3, 21, 7)
            res = simple_wiener(gray, kernel_size=ks)
            show_image_col(res, caption=f"Local variance filter (approx Wiener), k={ks}")

    elif module == "Morphology & Segmentation":
        st.write("Binary and grayscale morphology, thresholding, skeletonization.")
        gray = to_gray(img_rgb).astype(np.uint8)
        method = st.selectbox("Operation", [
            "Global Threshold (Otsu)",
            "Adaptive Threshold (Gaussian)",
            "Morphological ops: Erode / Dilate / Open / Close",
            "Skeletonization (binary)",
            "Edge detection (Sobel / Canny)"
        ])
        if method == "Global Threshold (Otsu)":
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            show_image_col(th, caption=f"Otsu threshold (value auto)")
        elif method == "Adaptive Threshold (Gaussian)":
            block = st.slider("Block size (odd)", 3, 51, 11, step=2)
            C = st.slider("C (subtracted)", 0, 20, 2)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, C)
            show_image_col(th, caption=f"Adaptive threshold block={block}, C={C}")
        elif method == "Morphological ops: Erode / Dilate / Open / Close":
            op = st.selectbox("Choose op", ["Erode", "Dilate", "Open", "Close", "Tophat", "Blackhat"])
            k = st.slider("Kernel size", 1, 31, 3, step=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            if op == "Erode":
                out = cv2.erode(gray, kernel, iterations=1)
            elif op == "Dilate":
                out = cv2.dilate(gray, kernel, iterations=1)
            elif op == "Open":
                out = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            elif op == "Close":
                out = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            elif op == "Tophat":
                out = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            else:
                out = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            show_image_col(out, caption=f"{op} (k={k})")
        elif method == "Skeletonization (binary)":
            # Binary first
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            inv = (th == 0).astype(np.uint8)  # skeletonize foreground as 1
            if SKIMAGE_AVAILABLE:
                sk = skeletonize(inv).astype(np.uint8) * 255
                show_image_col(sk, caption="Skeleton (skimage)")
            else:
                sk = naive_skeleton(th)
                show_image_col(sk, caption="Skeleton (naive fallback)")
        elif method == "Edge detection (Sobel / Canny)":
            e = st.selectbox("Edge method", ["Sobel", "Canny"])
            if e == "Sobel":
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                mag = np.hypot(sx, sy)
                mag = np.clip(mag / np.max(mag) * 255, 0, 255).astype(np.uint8)
                show_image_col(mag, caption="Sobel magnitude")
            else:
                t1 = st.slider("Canny low threshold", 10, 200, 50)
                t2 = st.slider("Canny high threshold", 50, 400, 150)
                edges = cv2.Canny(gray, t1, t2)
                show_image_col(edges, caption=f"Canny edges ({t1}, {t2})")

    elif module == "Compression (Block DCT / JPEG-like)":
        st.write("Block-DCT compression demo — demonstrates transform coding used in JPEG.")
        gray = to_gray(img_rgb).astype(np.uint8)
        block = st.slider("Block size", 4, 16, 8)
        quality = st.slider("Compression quality (1 worst, 100 best)", 1, 100, 50)
        compressed = block_dct_compress(gray, block=block, quality=quality)
        cols = st.columns(2)
        with cols[0]:
            show_image_col(gray, caption="Original (gray)")
        with cols[1]:
            show_image_col(compressed, caption=f"Block-DCT reconstructed (block={block}, Q={quality})")
        st.write("Observe blocking artifacts and loss as quality decreases. This demonstrates transform coding + quantization step in JPEG.")

    elif module == "Save / Download":
        st.write("Download processed image or snapshot.")
        # default processed is original; user can re-run other modules and then return to get that output.
        process_for_save = st.selectbox("Which image to save", ["Original (RGB)", "Gray", "Last processed by transform (use other tab first)"])
        if process_for_save == "Original (RGB)":
            data = pil_bytes_from_cv2(img_rgb)
            st.download_button("Download Original PNG", data=data, file_name="original.png", mime="image/png")
        elif process_for_save == "Gray":
            gray = to_gray(img_rgb).astype(np.uint8)
            is_success, buf = cv2.imencode(".png", gray)
            st.download_button("Download Gray PNG", data=buf.tobytes(), file_name="gray.png", mime="image/png")
        else:
            st.info("If you want to save a processed image, perform the operation first in another module, then come here and choose the appropriate option.")

# -------------------------
# Footer: tips & references
# -------------------------
st.markdown("---")
st.markdown("### Notes & tips")
st.markdown("""
- This app performs all computations locally on the image(s) you provide (camera or upload).  
- Use the **Sampling & Aliasing** demo to show how decimation without low-pass filtering causes aliasing (important for Unit III).  
- Use the **DFT / DCT** demonstrations for Unit II; the block-DCT compression mimics JPEG transform coding (Unit V).  
- **Homomorphic filtering** shows frequency-domain illumination/reflectance separation (Unit IV).  
- **Morphological** ops and skeletonization cover Unit V concepts: erosion, dilation, opening/closing, skeletonization.  
""")

st.markdown("© Image Processing Studio — built for FI1932 syllabus demos. Built with OpenCV & Streamlit.")
