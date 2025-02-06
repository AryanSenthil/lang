# color_frame_stream.py
import pyrealsense2 as rs
import numpy as np
from typing import Optional
from dataclasses import dataclass
import threading
import queue

@dataclass
class ColorFrame:
    """Data class to hold color frame data"""
    image: np.ndarray
    error: Optional[str] = None

class ColorRealSenseStream:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ColorRealSenseStream, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.pipe = None
        self.config = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.is_running = False
        self._initialize_pipeline()
        self._initialized = True
        
    def _initialize_pipeline(self):
        """Set up the RealSense pipeline"""
        try:
            self.pipe = rs.pipeline()
            self.config = rs.config()
            
            # Enable the color stream
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
            
            # Start pipeline with config
            self.pipe.start(self.config)

            # skip first 5 frames to allow auto-exposure to stabilize
            for _ in range(5):
                self.pipe.wait_for_frames()
                
            self.is_running = True
            
            # Start frame capture thread
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RealSense pipeline: {str(e)}")

    def _capture_frames(self):
        """Continuously capture frames in a separate thread"""
        while self.is_running:
            try:
                frames = self.pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if color_frame:
                    # Convert to numpy array
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Update queue with latest frame
                    try:
                        # Remove old frame if exists
                        if not self.frame_queue.empty():
                            self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(ColorFrame(image=color_image))
                    except queue.Full:
                        pass  # Skip frame if queue is full
                        
            except Exception as e:
                if self.is_running:  # Only log error if we're still supposed to be running
                    print(f"Error capturing frame: {str(e)}")

    def streaming_color_frame(self) -> ColorFrame:
        """Get the latest color frame"""
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return frame
        except queue.Empty:
            return ColorFrame(
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
                error="No frame available"
            )

    def stop(self):
        """Stop the RealSense pipeline"""
        self.is_running = False
        if self.pipe:
            self.pipe.stop()
            self._initialized = False