"""
Local OpenColorIO (OCIO) integration for Blender AgX
Exact matching with Blender's color management system
"""

import numpy as np
import os
import sys

_MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Check if PyOpenColorIO is installed
try:
    import PyOpenColorIO as OCIO
    OCIO_AVAILABLE = True
except ImportError:
    OCIO_AVAILABLE = False
    print("WARNING: PyOpenColorIO not found!")
    print("Install with: pip install opencolorio")
    print("Falling back to approximation...\n")


class BlenderAgX:
    """
    Wrapper for Blender's AgX color transform using OpenColorIO.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize with Blender's OCIO config.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to Blender's config.ocio file.
            If None, will try to auto-detect Blender installation.
        """
        if not OCIO_AVAILABLE:
            raise ImportError("PyOpenColorIO is required. Install with: pip install opencolorio")
        
        # Prefer the copy shipped with this repository so callers do not need
        # to change their working directory.  An explicit path still wins.
        self.config_path = config_path or os.path.join(_MODULE_ROOT, "config.ocio")
        
        if not self.config_path or not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Could not find config.ocio at: {self.config_path}\n"
                "Please provide the path to Blender's config.ocio file.\n"
                "Typical locations:\n"
                "  Linux: ~/.config/blender/4.x/datafiles/colormanagement/config.ocio\n"
                "  Windows: C:\\Program Files\\Blender Foundation\\Blender 4.x\\4.x\\datafiles\\colormanagement\\config.ocio\n"
                "  macOS: /Applications/Blender.app/Contents/Resources/4.x/datafiles/colormanagement/config.ocio"
            )
        
        # Load OCIO config
        self.config = OCIO.Config.CreateFromFile(self.config_path)
        print(f"✓ Loaded OCIO config from: {self.config_path}")
        
        # Available color spaces
        # self.list_available_spaces()
    
    def _find_blender_config(self):
        """Try to auto-detect Blender's config.ocio file."""
        possible_paths = []
        
        # Linux
        home = os.path.expanduser("~")
        for version in ["4.3", "4.2", "4.1", "4.0", "3.6"]:
            possible_paths.append(
                os.path.join(home, f".config/blender/{version}/datafiles/colormanagement/config.ocio")
            )
        
        # Windows
        for drive in ["C:", "D:"]:
            for version in ["4.3", "4.2", "4.1", "4.0", "3.6"]:
                possible_paths.append(
                    f"{drive}\\Program Files\\Blender Foundation\\Blender {version}\\{version}\\datafiles\\colormanagement\\config.ocio"
                )
        
        # macOS
        for version in ["4.3", "4.2", "4.1", "4.0", "3.6"]:
            possible_paths.append(
                f"/Applications/Blender.app/Contents/Resources/{version}/datafiles/colormanagement/config.ocio"
            )
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def list_available_spaces(self):
        """Print all available color spaces in the config."""
        print("\nAvailable color spaces:")
        for i in range(self.config.getNumColorSpaces()):
            cs = self.config.getColorSpaceNameByIndex(i)
            print(f"  - {cs}")
        print()
    
    def create_processor(self, src_space="Linear", dst_space="AgX Base", 
                         exposure=0.0, gamma=1.0):
        """
        Create a color processor from source to destination space.
        
        Parameters:
        -----------
        src_space : str
            Source color space (default: "Linear" or "Linear Rec.709")
        dst_space : str
            Destination color space. Options include:
            - "AgX Base" (default, no look)
            - "AgX Base Punchy" (increased saturation)
            - "AgX Base Greyscale"
            - "sRGB" (for standard transform)
        exposure : float
            Exposure adjustment in stops (applied before transform)
        gamma : float
            Gamma adjustment (applied after transform via display)
        
        Returns:
        --------
        OCIO.CPUProcessor
        """
        # Try different variations of "Linear" color space name
        linear_variants = ["Linear Rec.709", "Linear", "Linear CIE-XYZ E", "lin_rec709"]
        
        actual_src = None
        for variant in linear_variants:
            try:
                self.config.getColorSpace(variant)
                actual_src = variant
                break
            except:
                continue
        
        if actual_src is None:
            actual_src = src_space
        
        # Create the transform
        group_transform = OCIO.GroupTransform()
        
        # Add exposure adjustment if needed
        if exposure != 0.0:
            exposure_matrix = OCIO.MatrixTransform()
            scale = 2.0 ** exposure
            exposure_matrix.setMatrix([scale, 0, 0, 0,
                                      0, scale, 0, 0,
                                      0, 0, scale, 0,
                                      0, 0, 0, 1])
            group_transform.appendTransform(exposure_matrix)
        
        # Main color space transform
        cs_transform = OCIO.ColorSpaceTransform()
        cs_transform.setSrc(actual_src)
        cs_transform.setDst(dst_space)
        group_transform.appendTransform(cs_transform)
        
        # Add gamma adjustment if needed
        if gamma != 1.0:
            gamma_transform = OCIO.ExponentTransform()
            gamma_transform.setValue([1.0/gamma, 1.0/gamma, 1.0/gamma, 1.0])
            group_transform.appendTransform(gamma_transform)
        
        # Get processor
        processor = self.config.getProcessor(group_transform)
        return processor.getDefaultCPUProcessor()
    
    def apply(self, image, src_space="Linear", dst_space="AgX Base",
              exposure=0.0, gamma=1.0):
        """
        Apply AgX transform to an image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image, shape (H, W, 3) or (H, W, 4), dtype float32 or float64
        src_space : str
            Source color space (default: "Linear")
        dst_space : str
            Destination color space (default: "AgX Base")
        exposure : float
            Exposure adjustment in stops
        gamma : float
            Gamma correction
        
        Returns:
        --------
        np.ndarray
            Transformed image, same shape and dtype as input
        """
        # Validate input
        if image.ndim != 3 or image.shape[2] not in [3, 4]:
            raise ValueError("Image must be shape (H, W, 3) or (H, W, 4)")
        
        # Get processor
        cpu = self.create_processor(src_space, dst_space, exposure, gamma)
        
        # Prepare data
        original_dtype = image.dtype
        img = image.astype(np.float32).copy()
        
        # OCIO works on flattened arrays
        h, w, c = img.shape
        img_flat = img.reshape(-1, c)
        
        # Apply transform
        if c == 3:
            cpu.applyRGB(img_flat)
        else:  # RGBA
            cpu.applyRGBA(img_flat)
        
        # Reshape and restore dtype
        result = img_flat.reshape(h, w, c)
        
        if original_dtype != np.float32:
            result = result.astype(original_dtype)
        
        return result


# Fallback approximation if OCIO is not available
def agx_approximation(linear_rgb, exposure=0.0, gamma=1.0):
    """
    Simplified AgX approximation (does NOT match Blender exactly).
    Use this only if OpenColorIO is not available.
    """
    linear_rgb = linear_rgb * (2.0 ** exposure)
    linear_rgb = np.maximum(linear_rgb, 0.0)
    
    min_ev, max_ev = -10.0, 6.5
    epsilon = 1e-10
    log_rgb = np.log2(linear_rgb + epsilon)
    log_rgb = (log_rgb - min_ev) / (max_ev - min_ev)
    log_rgb = np.clip(log_rgb, 0.0, 1.0)
    
    # AgX sigmoid
    x = log_rgb
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    
    encoded = (
        - 17.86 * x6 * x +
        78.01 * x6 -
        126.7 * x4 * x +
        92.06 * x4 -
        28.72 * x2 * x +
        4.361 * x2 -
        0.1718 * x +
        0.002857
    )
    
    encoded = np.clip(encoded, 0.0, 1.0)
    
    if gamma != 1.0:
        encoded = np.power(encoded, 1.0 / gamma)
    
    return np.clip(encoded, 0.0, 1.0)

agx = BlenderAgX()


def _ocio_apply(image: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Apply an OCIO color space transform; alpha is passed through bit-exact."""
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected (H, W, 3) or (H, W, 4), got {image.shape}")
    original_dtype = image.dtype
    img = image.astype(np.float32)
    h, w, c = img.shape
    has_alpha = c == 4

    # always transform RGB only; restore alpha afterwards
    rgb = np.ascontiguousarray(img[..., :3]).reshape(-1, 3)
    proc = agx.config.getProcessor(OCIO.ColorSpaceTransform(src=src, dst=dst))
    cpu = proc.getDefaultCPUProcessor()
    cpu.applyRGB(rgb)

    result = np.empty_like(img)
    result[..., :3] = rgb.reshape(h, w, 3)
    if has_alpha:
        result[..., 3] = img[..., 3]   # bit-exact copy

    return result if original_dtype == np.float32 else result.astype(original_dtype)

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Blender AgX with OpenColorIO")
    print("="*70)
    
    if not OCIO_AVAILABLE:
        print("\n⚠️  OpenColorIO not installed!")
        print("Install with: pip install opencolorio")
        print("\nUsing approximation for demonstration...\n")
        
        # Demo with approximation
        test_img = np.random.rand(100, 100, 3).astype(np.float32) * 2.0
        result = agx_approximation(test_img)
        print(f"Approximation result: {result.shape}, range [{result.min():.3f}, {result.max():.3f}]")
    
    else:
        print("\n✓ OpenColorIO is installed!")
        print("\nTrying to initialize Blender AgX...\n")
        
        try:
            # Initialize (will auto-detect Blender installation)
            agx = BlenderAgX()
            
            # Create test image
            print("Creating test image...")
            test_img = np.random.rand(100, 100, 3).astype(np.float32) * 2.0
            
            # Apply AgX Base (default)
            print("\nApplying AgX Base transform...")
            result_base = agx.apply(test_img, dst_space="AgX Base sRGB")
            print(f"✓ Result: {result_base.shape}, range [{result_base.min():.3f}, {result_base.max():.3f}]")
            breakpoint()
            
            # Apply with exposure adjustment
            print("\nApplying AgX with +1 stop exposure...")
            result_bright = agx.apply(test_img, dst_space="AgX Base", exposure=1.0)
            print(f"✓ Result: {result_bright.shape}, range [{result_bright.min():.3f}, {result_bright.max():.3f}]")
            
            # Try different looks
            print("\nTrying AgX Base Punchy...")
            try:
                result_punchy = agx.apply(test_img, dst_space="AgX Base Punchy")
                print(f"✓ Result: {result_punchy.shape}")
            except:
                print("  (AgX Base Punchy not available in this config)")
            
            print("\n" + "="*70)
            print("✓ Setup complete! You can now use:")
            print("  agx = BlenderAgX()")
            print("  result = agx.apply(your_image, exposure=0.0, gamma=1.0)")
            print("="*70)
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("\nPlease provide the path manually:")
            print("  agx = BlenderAgX(config_path='/path/to/config.ocio')")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
