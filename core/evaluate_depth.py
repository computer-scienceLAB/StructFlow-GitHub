"""Depth/disparity-only evaluation entry point; never exports optical flow."""
from reproduce.evaluate_task import main

if __name__ == '__main__':
    main('depth')
