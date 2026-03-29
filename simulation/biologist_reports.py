#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOGOS-CA Example: Fabricated Biological Reports

This example is based on examples/alife.py, but uses the modified logos_ca.py
that expects direct JSON reports with the following keys:
- title
- summary
- sketch_rgb_16x16

In this setup, all cells start from the same neutral placeholder report.
At each step, the LLM is asked to write an English report based on the
previous local report and the nearby reports. Over time, reports may drift,
spread, merge, and diversify across the grid.

Usage:
    $ python biologist_reports.py
"""

import sys
import os
import json

# Add parent directory to path for importing logos_ca
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from logos_ca import LOGOSConfig, LOGOSSimulator


# =============================================================================
# Configuration
# =============================================================================

# Create configuration for fabricated report simulation
# Adjust these parameters according to your needs and API rate limits
config = LOGOSConfig(
    # Grid dimensions
    grid_size_x=10,
    grid_size_y=10,

    # Number of simulation steps
    max_steps=5,

    # LLM model selection
    model_name="gpt-5-mini",

    # Temperature for LLM sampling
    temperature=1.0,

    # Parallel processing settings
    max_workers=10,
    delay_between_calls=1.0,

    # Retry settings for API failures
    max_retries=10,
    retry_delay=5.0,

    # Output file for results
    output_json="biologist_reports_result.json",

    # Enable resume to continue interrupted simulations
    enable_resume=True,
)


# =============================================================================
# Grid Initialization
# =============================================================================


def initialize_grid(size_x: int, size_y: int) -> np.ndarray:
    """
    Initialize the grid for fabricated report simulation.

    All cells start from the same neutral placeholder report.

    Args:
        size_x: Number of cells in the horizontal direction.
        size_y: Number of cells in the vertical direction.

    Returns:
        2D numpy array of cell descriptions with dtype=object.
    """
    # Create empty grid
    grid = np.empty((size_y, size_x), dtype=object)

    # Fill all cells with the same neutral placeholder report
    empty_state = json.dumps(
        {
            "title": "Unknown",
            "summary": "Unavailable.",
            "sketch_rgb_16x16": [
                [[0, 0, 0] for _ in range(16)]
                for _ in range(16)
            ],
        },
        ensure_ascii=False,
    )

    for i in range(size_y):
        for j in range(size_x):
            grid[i, j] = empty_state

    return grid


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Create simulator with the configuration and initializer
    simulator = LOGOSSimulator(config, initialize_grid)

    # Run the simulation
    try:
        history = simulator.run()

        print("\n" + "=" * 70)
        print("🎉 Fabricated report simulation completed successfully!")
        print(f"   Total steps: {len(history)}")
        print(f"   Results saved to: {config.output_json}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user.")
        print("   Progress has been saved. Run again to resume.")
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        raise
