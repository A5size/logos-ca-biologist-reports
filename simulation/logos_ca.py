#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOGOS-CA: Language Oriented Grid Of Statements - Cellular Automaton

A framework for running cellular automata where cell states and update rules
are described in natural language, with state transitions determined by a
Large Language Model (LLM).

This module provides the core functionality for LOGOS-CA simulations.
Users should import LOGOSConfig and LOGOSSimulator to create and run
their own simulations.

API Note:
    This implementation uses OpenAI's Responses API, which is the recommended
    API for new projects as of 2025. The Responses API provides better
    performance, lower costs through improved cache utilization, and cleaner
    semantics compared to the legacy Chat Completions API.

Reference:
    Utimula, K. (2026). LOGOS-CA: A Cellular Automaton Using Natural Language
    as State and Rule. arXiv preprint.

Example:
    >>> from logos_ca import LOGOSConfig, LOGOSSimulator
    >>> import numpy as np
    >>>
    >>> config = LOGOSConfig(grid_size_x=10, grid_size_y=10, max_steps=5)
    >>>
    >>> def my_initializer(size_x, size_y):
    ...     grid = np.empty((size_y, size_x), dtype=object)
    ...     for i in range(size_y):
    ...         for j in range(size_x):
    ...             grid[i, j] = "Empty cell"
    ...     grid[size_y // 2, size_x // 2] = "Active cell that spreads"
    ...     return grid
    >>>
    >>> simulator = LOGOSSimulator(config, my_initializer)
    >>> simulator.run()
"""

import json
import re
import time
import threading
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI


# =============================================================================
# Exceptions
# =============================================================================

class LOGOSSimulationError(Exception):
    """
    Exception raised when the simulation cannot continue due to API failures.
    
    This exception is raised when a cell update fails after exhausting all
    retry attempts. The simulation is halted to ensure data integrity, as
    continuing with partially updated grids would compromise the validity
    of the simulation results.
    
    Attributes:
        cell_pos: Tuple of (row, column) indicating which cell failed.
        message: Description of the failure.
    """
    
    def __init__(self, cell_pos: Tuple[int, int], message: str):
        self.cell_pos = cell_pos
        self.message = message
        super().__init__(f"Cell {cell_pos}: {message}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LOGOSConfig:
    """
    Configuration class for LOGOS-CA simulations.
    
    This dataclass holds all parameters needed to run a LOGOS-CA simulation.
    Users can customize grid size, number of steps, LLM model, parallel
    processing settings, and output options.
    
    Attributes:
        grid_size_x (int): Number of cells in the horizontal direction.
            Default is 25.
        grid_size_y (int): Number of cells in the vertical direction.
            Default is 25.
        max_steps (int): Maximum number of simulation steps to run.
            Default is 100.
        model_name (str): Name of the OpenAI model to use for state updates.
            Default is "gpt-4o".
        temperature (float): Temperature parameter for LLM sampling.
            Higher values increase randomness. Default is 1.0.
        max_workers (int): Maximum number of parallel threads for API calls.
            Adjust based on API rate limits. Default is 10.
        delay_between_calls (float): Delay in seconds between API calls
            to avoid rate limiting. Default is 1.0.
        max_retries (int): Maximum number of retry attempts for failed
            API calls. If all retries fail, the simulation is halted.
            Default is 10.
        retry_delay (float): Delay in seconds between retry attempts.
            Default is 5.0.
        output_json (str): Path to the output JSON file for saving results.
            Default is "simulation_result.json".
        enable_resume (bool): If True, attempts to resume from existing
            simulation results. Default is True.
        api_key (Optional[str]): OpenAI API key. If None, reads from
            OPENAI_API_KEY environment variable. Default is None.
    
    Example:
        >>> config = LOGOSConfig(
        ...     grid_size_x=11,
        ...     grid_size_y=11,
        ...     max_steps=10,
        ...     model_name="gpt-4o",
        ...     output_json="my_simulation.json"
        ... )
    """
    # Grid dimensions
    grid_size_x: int = 25
    grid_size_y: int = 25
    
    # Simulation parameters
    max_steps: int = 100
    
    # LLM settings
    model_name: str = "gpt-4o"
    temperature: float = 1.0
    
    # Parallel processing settings
    max_workers: int = 10
    delay_between_calls: float = 1.0
    
    # Retry settings for failed API calls
    # If all retries fail, the simulation is halted to ensure data integrity
    max_retries: int = 10
    retry_delay: float = 5.0
    
    # Output settings
    output_json: str = "simulation_result.json"
    enable_resume: bool = True
    
    # API key (if None, will use environment variable)
    api_key: Optional[str] = None


# =============================================================================
# Pricing Information
# =============================================================================

# Pricing table for OpenAI models (USD per 1 million tokens)
# Update these values as pricing changes
MODEL_PRICES_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-5":       {"input": 1.25, "output": 10.00},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00},
    "gpt-5-nano":  {"input": 0.05, "output": 0.40},
}


# =============================================================================
# Usage Tracking
# =============================================================================

class UsageTracker:
    """
    Thread-safe tracker for API usage statistics.
    
    This class maintains cumulative counts of API calls and token usage
    across all parallel threads during a simulation run.
    
    Attributes:
        calls (int): Total number of API calls made.
        prompt_tokens (int): Total number of input tokens used.
        completion_tokens (int): Total number of output tokens generated.
        total_tokens (int): Total tokens (input + output).
    
    Example:
        >>> tracker = UsageTracker()
        >>> tracker.record({"input_tokens": 100, "output_tokens": 50})
        >>> print(tracker.get_stats())
    """
    
    def __init__(self):
        """Initialize the usage tracker with zero counts."""
        self._lock = threading.Lock()
        self._stats = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    
    def record(self, usage: Dict[str, int]) -> None:
        """
        Record usage statistics from an API response.
        
        This method is thread-safe and can be called from multiple threads
        simultaneously.
        
        Args:
            usage: Dictionary containing token usage information.
                Expected keys are "input_tokens" and "output_tokens".
        """
        if not usage:
            return
        
        # Extract token counts from usage object
        in_tok = usage.get("input_tokens", 0) or 0
        out_tok = usage.get("output_tokens", 0) or 0
        tot_tok = usage.get("total_tokens", 0) or (in_tok + out_tok)
        
        with self._lock:
            self._stats["calls"] += 1
            self._stats["input_tokens"] += in_tok
            self._stats["output_tokens"] += out_tok
            self._stats["total_tokens"] += tot_tok
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get a copy of current usage statistics.
        
        Returns:
            Dictionary containing current usage counts.
        """
        with self._lock:
            return self._stats.copy()
    
    def calculate_cost(self, model_name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate estimated cost based on usage and model pricing.
        
        Args:
            model_name: Name of the model to look up pricing for.
        
        Returns:
            Tuple of (total_cost, input_cost, output_cost) in USD.
            Returns (None, None, None) if model pricing is not available.
        """
        price = MODEL_PRICES_USD_PER_1M.get(model_name)
        if not price:
            return None, None, None
        
        stats = self.get_stats()
        in_cost = (stats["input_tokens"] / 1_000_000) * price["input"]
        out_cost = (stats["output_tokens"] / 1_000_000) * price["output"]
        return in_cost + out_cost, in_cost, out_cost


class ErrorTracker:
    """
    Thread-safe tracker for error and success statistics.
    
    This class monitors the success rate of cell updates, tracking
    first-attempt successes, retries, and failures.
    
    Attributes:
        total_cells_processed (int): Total number of cells processed.
        successful_first_try (int): Cells updated successfully on first attempt.
        successful_after_retry (int): Cells updated successfully after retrying.
        api_errors (int): Total API call errors encountered.
        json_extraction_errors (int): Total JSON parsing errors encountered.
    """
    
    def __init__(self):
        """Initialize the error tracker with zero counts."""
        self._lock = threading.Lock()
        self._stats = {
            "total_cells_processed": 0,
            "successful_first_try": 0,
            "successful_after_retry": 0,
            "api_errors": 0,
            "json_extraction_errors": 0,
            "sketch_rgb_16x16_failures": 0,
            "sketch_rgb_16x16_normalizations": 0,
        }
    
    def record(self, error_type: str, count: int = 1) -> None:
        """
        Record an error or success event.
        
        Args:
            error_type: Type of event to record. Must be one of the
                tracked statistics.
            count: Number of events to record. Default is 1.
        """
        with self._lock:
            if error_type in self._stats:
                self._stats[error_type] += count
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get a copy of current error statistics.
        
        Returns:
            Dictionary containing current error and success counts.
        """
        with self._lock:
            return self._stats.copy()
    
    def print_summary(self) -> None:
        """
        Print a formatted summary of error statistics.
        """
        stats = self.get_stats()
        
        print("\n" + "=" * 70)
        print("📊 PROCESSING STATISTICS")
        print("=" * 70)
        
        total = stats["total_cells_processed"]
        successful = stats["successful_first_try"] + stats["successful_after_retry"]
        
        print(f"\n🔢 Processing Summary:")
        print(f"   Total cells processed: {total}")
        if total > 0:
            print(f"   Successful updates: {successful} ({successful/total*100:.2f}%)")
        
        print(f"\n✅ Success Breakdown:")
        if total > 0:
            print(f"   First attempt success: {stats['successful_first_try']} "
                  f"({stats['successful_first_try']/total*100:.2f}%)")
            print(f"   Success after retry: {stats['successful_after_retry']} "
                  f"({stats['successful_after_retry']/total*100:.2f}%)")
        
        print(f"\n⚠️  Error Counts (recovered via retry):")
        print(f"   API call errors: {stats['api_errors']}")
        print(f"   JSON extraction errors: {stats['json_extraction_errors']}")
        
        print(f"\n🖼️  Sketch Statistics:")
        print(f"   Invalid sketch outputs: {stats['sketch_rgb_16x16_failures']}")
        print(f"   Sketch normalizations applied: {stats['sketch_rgb_16x16_normalizations']}")
        
        print("=" * 70)


# =============================================================================
# Utility Functions
# =============================================================================

def extract_json_from_response(llm_output: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON content from LLM output text.
    
    This function attempts to parse JSON from the LLM response, handling
    direct JSON, markdown code blocks (```json ... ```), and raw JSON objects.
    
    Args:
        llm_output: Raw text output from the LLM.
    
    Returns:
        Parsed JSON as a dictionary if successful, None otherwise.
    """
    if not llm_output:
        return None
    
    llm_output = llm_output.strip()
    
    # First, try parsing the entire response as JSON
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass
    
    # Next, try to extract JSON from markdown code blocks
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    
    # If no code blocks found, try to find raw JSON objects
    if not matches:
        json_pattern = r"\{.*\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)
    
    # Try to parse each match
    for json_string in matches:
        json_string = json_string.strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            # Try cleaning control characters and retry
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                return json.loads(json_string_clean)
            except json.JSONDecodeError:
                continue
    
    return None


def get_moore_neighborhood(grid: np.ndarray, i: int, j: int) -> List[Dict[str, str]]:
    """
    Get the Moore neighborhood (8 surrounding cells) for a given cell.
    
    The Moore neighborhood consists of the 8 cells surrounding a central cell
    in a 2D grid. This function implements wrap-around (toroidal) boundary
    conditions, meaning the grid wraps at the edges.
    
    Args:
        grid: 2D numpy array of cell descriptions.
        i: Row index of the target cell.
        j: Column index of the target cell.
    
    Returns:
        List of dictionaries, each containing:
            - "position": Direction label (e.g., "top-left", "bottom")
            - "description": Cell state description at that position
    
    Example:
        >>> grid = np.array([["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]])
        >>> neighbors = get_moore_neighborhood(grid, 1, 1)  # Center cell "E"
        >>> for n in neighbors:
        ...     print(f"{n['position']}: {n['description']}")
    """
    size_y, size_x = grid.shape
    neighbors = []
    
    # Direction offsets and their labels
    # Format: (row_offset, col_offset, label)
    directions = [
        (-1, -1, "top-left"),
        (-1,  0, "top"),
        (-1,  1, "top-right"),
        ( 0, -1, "left"),
        ( 0,  1, "right"),
        ( 1, -1, "bottom-left"),
        ( 1,  0, "bottom"),
        ( 1,  1, "bottom-right"),
    ]
    
    for di, dj, label in directions:
        # Apply wrap-around boundary conditions
        ni = (i + di) % size_y
        nj = (j + dj) % size_x
        neighbors.append({
            "position": label,
            "description": grid[ni, nj]
        })
    
    return neighbors


def get_von_neumann_neighborhood(grid: np.ndarray, i: int, j: int) -> List[Dict[str, str]]:
    """
    Get the von Neumann neighborhood (4 orthogonally adjacent cells)
    for a given cell.

    This function implements wrap-around (toroidal) boundary conditions
    over the entire grid.

    Args:
        grid: 2D numpy array of cell descriptions.
        i: Row index of the target cell.
        j: Column index of the target cell.

    Returns:
        List of dictionaries, each containing:
            - "position": Direction label (e.g., "top", "left")
            - "description": Cell state description at that position
    """
    size_y, size_x = grid.shape
    neighbors = []

    directions = [
        (-1,  0, "top"),
        ( 0, -1, "left"),
        ( 0,  1, "right"),
        ( 1,  0, "bottom"),
    ]

    for di, dj, label in directions:
        ni = (i + di) % size_y
        nj = (j + dj) % size_x
        neighbors.append({
            "position": label,
            "description": grid[ni, nj]
        })

    return neighbors


def get_sparse_region_von_neumann_neighborhood(grid: np.ndarray, i: int, j: int) -> List[Dict[str, str]]:
    """
    Get the custom von Neumann neighborhood for a 10x10 grid divided into
    four weakly coupled 5x5 regions.

    Within each 5x5 region, boundary conditions are toroidal. Only the
    following cross-region connections are allowed:
        1 and 2: (2, 4) <-> (2, 5)
        1 and 3: (4, 2) <-> (5, 2)
        2 and 4: (4, 7) <-> (5, 7)
        3 and 4: (7, 4) <-> (7, 5)

    Args:
        grid: 2D numpy array of cell descriptions.
        i: Row index of the target cell.
        j: Column index of the target cell.

    Returns:
        List of dictionaries, each containing:
            - "position": Direction label (e.g., "top", "left")
            - "description": Cell state description at that position
    """
    size_y, size_x = grid.shape
    if size_y != 10 or size_x != 10:
        raise ValueError(
            "get_sparse_region_von_neumann_neighborhood expects a 10x10 grid."
        )

    cross_region_links = {
        (2, 4, "right"): (2, 5),
        (2, 5, "left"): (2, 4),
        (4, 2, "bottom"): (5, 2),
        (5, 2, "top"): (4, 2),
        (4, 7, "bottom"): (5, 7),
        (5, 7, "top"): (4, 7),
        (7, 4, "right"): (7, 5),
        (7, 5, "left"): (7, 4),
    }

    region_row = 0 if i < 5 else 1
    region_col = 0 if j < 5 else 1
    row_start = region_row * 5
    col_start = region_col * 5

    local_i = i - row_start
    local_j = j - col_start

    neighbors = []
    directions = [
        (-1,  0, "top"),
        ( 0, -1, "left"),
        ( 0,  1, "right"),
        ( 1,  0, "bottom"),
    ]

    for di, dj, label in directions:
        if (i, j, label) in cross_region_links:
            ni, nj = cross_region_links[(i, j, label)]
        else:
            ni = row_start + ((local_i + di) % 5)
            nj = col_start + ((local_j + dj) % 5)

        neighbors.append({
            "position": label,
            "description": grid[ni, nj]
        })

    return neighbors


def grid_to_dict(grid: np.ndarray) -> Dict[str, str]:
    """
    Convert a 2D numpy grid to a dictionary for JSON serialization.
    
    Args:
        grid: 2D numpy array of cell descriptions.
    
    Returns:
        Dictionary mapping "row,col" string keys to cell descriptions.
    
    Example:
        >>> grid = np.array([["A", "B"], ["C", "D"]])
        >>> grid_to_dict(grid)
        {'0,0': 'A', '0,1': 'B', '1,0': 'C', '1,1': 'D'}
    """
    size_y, size_x = grid.shape
    grid_dict = {}
    for i in range(size_y):
        for j in range(size_x):
            grid_dict[f"{i},{j}"] = grid[i, j]
    return grid_dict


def dict_to_grid(grid_dict: Dict[str, str], size_x: int, size_y: int) -> np.ndarray:
    """
    Convert a dictionary back to a 2D numpy grid.
    
    Args:
        grid_dict: Dictionary mapping "row,col" keys to cell descriptions.
        size_x: Number of columns in the grid.
        size_y: Number of rows in the grid.
    
    Returns:
        2D numpy array of cell descriptions.
    
    Example:
        >>> d = {'0,0': 'A', '0,1': 'B', '1,0': 'C', '1,1': 'D'}
        >>> dict_to_grid(d, 2, 2)
        array([['A', 'B'],
               ['C', 'D']], dtype=object)
    """
    grid = np.empty((size_y, size_x), dtype=object)
    for key, value in grid_dict.items():
        i, j = map(int, key.split(','))
        grid[i, j] = value
    return grid


# =============================================================================
# Main Simulator Class
# =============================================================================

class LOGOSSimulator:
    """
    Main simulator class for running LOGOS-CA simulations.
    
    This class orchestrates the entire simulation process, including:
    - Grid initialization using a user-provided function
    - Parallel LLM queries for cell state updates
    - Progress tracking and cost estimation
    - Result saving and resume functionality
    
    The simulator uses OpenAI's Responses API for LLM inference, which
    provides better performance and lower costs compared to the legacy
    Chat Completions API.
    
    Error Handling:
        If a cell update fails after exhausting all retry attempts, the
        simulation is halted immediately. This ensures data integrity,
        as continuing with partially updated grids would compromise the
        validity of the simulation results. The progress up to the failure
        point is saved, allowing investigation and potential resumption
        after resolving the issue.
    
    Attributes:
        config (LOGOSConfig): Simulation configuration.
        initializer (Callable): Function to initialize the grid.
        history (List[np.ndarray]): List of grid states at each step.
        usage_tracker (UsageTracker): API usage statistics.
        error_tracker (ErrorTracker): Error and success statistics.
    
    Example:
        >>> config = LOGOSConfig(grid_size_x=10, grid_size_y=10, max_steps=5)
        >>> def init_func(size_x, size_y):
        ...     grid = np.empty((size_y, size_x), dtype=object)
        ...     for i in range(size_y):
        ...         for j in range(size_x):
        ...             grid[i, j] = "Empty"
        ...     return grid
        >>> sim = LOGOSSimulator(config, init_func)
        >>> sim.run()
    """
    
    # System prompt (instructions) for LLM queries
    SYSTEM_INSTRUCTIONS = """You are a biologist who was unable to conduct the survey personally, but you must still submit an original and novel short report. Fortunately, you can consult past reports from your assigned area and from nearby areas, so use them as references and compose an interesting fictional report. The report must not make it obvious that you did not actually observe the subject yourself. It should include a title, a brief summary strictly limited to 300 words or fewer, and a sketch of the organism you discovered (or of the environment, biome, or other subject of the report).

Return only JSON in English with this format:
{
  "title": "...",
  "summary": "...",
  "sketch_rgb_16x16": [[[R,G,B], ...], ...]
}

The sketch must be 16x16, and each pixel must be an RGB triplet of integers from 0 to 255.
"""
    
    # User prompt template for LLM queries
    USER_PROMPT_TEMPLATE = """Previous report:
{target_cell}

Nearby reports:
{neighbor_desc}

Write the next report.
"""

    # Early-phase system prompt for LLM queries
    EARLY_PHASE_SYSTEM_INSTRUCTIONS = """You are a biologist who was unable to conduct the survey personally, but you must still submit an original and novel short report. Fortunately, you can consult several past reports from regions fundamentally different from your assigned area. Those reports do not directly indicate your local conditions. The organism, environment, biome, or other subject described in your report must be fundamentally different from those described in the referenced reports. Do not reuse their defining features, central motifs, or characteristic elements. Compose an interesting fictional report that does not make it obvious that you did not actually observe the subject yourself. It should include a title, a brief summary strictly limited to 300 words or fewer, and a sketch of the organism you discovered (or of the environment, biome, or other subject of the report).

Return only JSON in English with this format:
{
  "title": "...",
  "summary": "...",
  "sketch_rgb_16x16": [[[R,G,B], ...], ...]
}

The sketch must be 16x16, and each pixel must be an RGB triplet of integers from 0 to 255.
"""

    # Early-phase user prompt template for LLM queries
    EARLY_PHASE_USER_PROMPT_TEMPLATE = """Reports from environmentally different regions:
{neighbor_desc}

Write the next report about a fundamentally different subject.
"""

    # Number of early steps to apply the early-phase prompts
    EARLY_PHASE_STEPS = 0

    # Seed generation settings for the weakly coupled 4-region setup
    SEED_GRID_SIZE_X = 5
    SEED_GRID_SIZE_Y = 5
    SEED_GENERATION_STEPS = 5
    REGION_SOURCE_STEPS = (1, 2, 3, 4)
    
    def __init__(self, config: LOGOSConfig, initializer: Callable[[int, int], np.ndarray]):
        """
        Initialize the LOGOS-CA simulator.
        
        Args:
            config: Configuration object containing simulation parameters.
            initializer: Function that takes (size_x, size_y) and returns
                a 2D numpy array of initial cell descriptions.
        """
        self.config = config
        self.initializer = initializer
        self.history: List[np.ndarray] = []
        self.usage_tracker = UsageTracker()
        self.error_tracker = ErrorTracker()
        
        # Set up OpenAI client
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key) if api_key else None
    
    def _get_system_instructions_for_step(self, step: int) -> str:
        """
        Get the system instructions for a given simulation step.
        
        Args:
            step: Current simulation step number.
        
        Returns:
            System instruction string for the step.
        """
        if step <= self.EARLY_PHASE_STEPS:
            return self.EARLY_PHASE_SYSTEM_INSTRUCTIONS
        return self.SYSTEM_INSTRUCTIONS

    def _get_user_prompt_template_for_step(self, step: int) -> str:
        """
        Get the user prompt template for a given simulation step.
        
        Args:
            step: Current simulation step number.
        
        Returns:
            User prompt template string for the step.
        """
        if step <= self.EARLY_PHASE_STEPS:
            return self.EARLY_PHASE_USER_PROMPT_TEMPLATE
        return self.USER_PROMPT_TEMPLATE
    
    @staticmethod
    def _is_valid_sketch_16x16(sketch: Any) -> bool:
        """
        Validate that a sketch is exactly a 16x16 grid of RGB triplets.
        
        Args:
            sketch: Sketch object to validate.
        
        Returns:
            True if the sketch matches the expected schema, False otherwise.
        """
        if not isinstance(sketch, list) or len(sketch) != 16:
            return False
        
        for row in sketch:
            if not isinstance(row, list) or len(row) != 16:
                return False
            for pixel in row:
                if not isinstance(pixel, list) or len(pixel) != 3:
                    return False
                if not all(isinstance(c, int) and 0 <= c <= 255 for c in pixel):
                    return False
        
        return True
    
    @staticmethod
    def _normalize_sketch_16x16(sketch: Any) -> List[List[List[int]]]:
        """
        Normalize a sketch to exactly 16x16 pixels.
        
        Invalid or missing rows/pixels are replaced with black pixels.
        Extra rows/pixels are truncated.
        
        Args:
            sketch: Sketch object to normalize.
        
        Returns:
            A normalized 16x16 RGB sketch.
        """
        black = [0, 0, 0]
        
        if not isinstance(sketch, list):
            sketch = []
        
        normalized_rows: List[List[List[int]]] = []
        for row in sketch[:16]:
            if not isinstance(row, list):
                row = []
            
            normalized_row: List[List[int]] = []
            for pixel in row[:16]:
                if (
                    isinstance(pixel, list)
                    and len(pixel) == 3
                    and all(isinstance(c, int) and 0 <= c <= 255 for c in pixel)
                ):
                    normalized_row.append([pixel[0], pixel[1], pixel[2]])
                else:
                    normalized_row.append(black.copy())
            
            while len(normalized_row) < 16:
                normalized_row.append(black.copy())
            
            normalized_rows.append(normalized_row)
        
        while len(normalized_rows) < 16:
            normalized_rows.append([black.copy() for _ in range(16)])
        
        return normalized_rows
    
    @classmethod
    def _normalize_report_json(cls, data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Normalize a parsed report JSON object.
        
        Args:
            data: Parsed JSON response from the LLM.
        
        Returns:
            Tuple of (normalized report dict or None, whether sketch normalization
            was applied).
        """
        if not isinstance(data, dict):
            return None, False
        
        if set(data.keys()) != {"title", "summary", "sketch_rgb_16x16"}:
            return None, False
        
        title = data["title"]
        summary = data["summary"]
        if not isinstance(title, str) or not isinstance(summary, str):
            return None, False
        
        sketch = data["sketch_rgb_16x16"]
        sketch_was_invalid = not cls._is_valid_sketch_16x16(sketch)
        normalized = {
            "title": title,
            "summary": summary,
            "sketch_rgb_16x16": cls._normalize_sketch_16x16(sketch),
        }
        return normalized, sketch_was_invalid
    
    @classmethod
    def _normalize_cell_state_text(cls, cell_state: str) -> str:
        """
        Normalize a serialized cell state before reusing it in prompts.
        
        This ensures previously generated sketches are always passed to the LLM
        as valid 16x16 arrays, even when resuming from older outputs.
        
        Args:
            cell_state: Serialized cell state string.
        
        Returns:
            Normalized serialized cell state string when possible, otherwise the
            original input string.
        """
        if not isinstance(cell_state, str):
            return str(cell_state)
        
        try:
            parsed = json.loads(cell_state)
        except Exception:
            return cell_state
        
        normalized, _ = cls._normalize_report_json(parsed)
        if normalized is None:
            return cell_state
        
        return json.dumps(normalized, ensure_ascii=False)
    
    @staticmethod
    def _is_valid_report_json(data: Dict[str, Any]) -> bool:
        """
        Validate that an LLM response matches the expected report schema.
        
        Args:
            data: Parsed JSON response from the LLM.
        
        Returns:
            True if the response matches the expected schema, False otherwise.
        """
        if not isinstance(data, dict):
            print("DEBUG VALIDATION: top-level is not a dict")
            return False
        
        expected_keys = {"title", "summary", "sketch_rgb_16x16"}
        actual_keys = set(data.keys())
        if actual_keys != expected_keys:
            print(f"DEBUG VALIDATION: keys mismatch: {actual_keys}")
            return False
        
        if not isinstance(data["title"], str):
            print(f"DEBUG VALIDATION: title is not str: {type(data['title'])}")
            return False
        
        if not isinstance(data["summary"], str):
            print(f"DEBUG VALIDATION: summary is not str: {type(data['summary'])}")
            return False
        
        sketch = data["sketch_rgb_16x16"]
        if not isinstance(sketch, list):
            print(f"DEBUG VALIDATION: sketch is not list: {type(sketch)}")
            return False
        
        if len(sketch) != 16:
            print(f"DEBUG VALIDATION: sketch row count is {len(sketch)}, expected 16")
            return False
        
        for i, row in enumerate(sketch):
            if not isinstance(row, list):
                print(f"DEBUG VALIDATION: row {i} is not list: {type(row)}")
                return False
            
            if len(row) != 16:
                print(f"DEBUG VALIDATION: row {i} length is {len(row)}, expected 16")
                return False
            
            for j, pixel in enumerate(row):
                if not isinstance(pixel, list):
                    print(f"DEBUG VALIDATION: pixel ({i},{j}) is not list: {type(pixel)}")
                    return False
                
                if len(pixel) != 3:
                    print(f"DEBUG VALIDATION: pixel ({i},{j}) length is {len(pixel)}, expected 3")
                    return False
                
                for k, c in enumerate(pixel):
                    if not isinstance(c, int):
                        print(f"DEBUG VALIDATION: pixel ({i},{j}) channel {k} is not int: {type(c)} value={c}")
                        return False
                    if not (0 <= c <= 255):
                        print(f"DEBUG VALIDATION: pixel ({i},{j}) channel {k} out of range: {c}")
                        return False
        
        return True
    
    def _query_llm_for_next_state(
        self,
        target_cell: str,
        neighbors: List[Dict[str, str]],
        cell_pos: Tuple[int, int],
        current_step: int
    ) -> Tuple[Tuple[int, int], str]:
        """
        Query the LLM to determine the next state of a cell.
        
        This method handles API calls with retry logic and error tracking.
        If all retry attempts fail, a LOGOSSimulationError is raised to
        halt the simulation.
        
        Args:
            target_cell: Current description of the target cell.
            neighbors: List of neighbor descriptions from the configured neighborhood function.
            cell_pos: (row, col) tuple identifying the cell position.
            current_step: Current simulation step number.
        
        Returns:
            Tuple of (cell_pos, next_state_description).
        
        Raises:
            LOGOSSimulationError: If the cell update fails after all retries.
        """
        # Normalize cell states before reusing them in prompts
        normalized_target_cell = self._normalize_cell_state_text(target_cell)
        normalized_neighbors = [
            {
                "position": n["position"],
                "description": self._normalize_cell_state_text(n["description"]),
            }
            for n in neighbors
        ]
        
        # Format neighbor descriptions for the prompt
        neighbor_desc = "\n".join([
            f"- {n['position']}: {n['description']}"
            for n in normalized_neighbors
        ])
        
        user_prompt = self._get_user_prompt_template_for_step(current_step).format(
            target_cell=normalized_target_cell,
            neighbor_desc=neighbor_desc
        )
        
        # Track this cell
        self.error_tracker.record("total_cells_processed")
        retry_count = 0
        last_error_message = ""
        
        # Retry loop
        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting delay
                time.sleep(self.config.delay_between_calls)
                
                # Make API call using OpenAI Responses API
                # The Responses API is recommended for new projects and provides
                # better performance and lower costs through improved caching
                response = self._client.responses.create(
                    model=self.config.model_name,
                    instructions=self._get_system_instructions_for_step(current_step),
                    input=user_prompt,
                    temperature=self.config.temperature,
                )
                
                # Record usage statistics
                if response.usage:
                    self.usage_tracker.record({
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.total_tokens,
                    })
                
                # Extract result from response
                # The Responses API provides output_text as a convenient accessor
                output_text = response.output_text
                result = extract_json_from_response(output_text)
                
                if result is None:
                    print(f"\n===== RAW LLM RESPONSE for cell {cell_pos}, attempt {attempt + 1} =====")
                    print(output_text)
                    print("===== END RAW LLM RESPONSE =====\n")
                    print(f"  DEBUG: extract_json_from_response returned None for cell {cell_pos}")
                    
                    self.error_tracker.record("json_extraction_errors")
                    retry_count += 1
                    last_error_message = f"JSON extraction failed. Response: {output_text[:200]}..."
                else:
                    normalized_result, sketch_was_invalid = self._normalize_report_json(result)
                    
                    if sketch_was_invalid:
                        self.error_tracker.record("sketch_rgb_16x16_failures")
                    
                    if normalized_result and self._is_valid_report_json(normalized_result):
                        if sketch_was_invalid:
                            self.error_tracker.record("sketch_rgb_16x16_normalizations")
                        
                        # Success
                        if attempt == 0:
                            self.error_tracker.record("successful_first_try")
                        else:
                            self.error_tracker.record("successful_after_retry")
                        return cell_pos, json.dumps(normalized_result, ensure_ascii=False)
                    
                    print(f"\n===== RAW LLM RESPONSE for cell {cell_pos}, attempt {attempt + 1} =====")
                    print(output_text)
                    print("===== END RAW LLM RESPONSE =====\n")
                    print(f"  DEBUG: extracted JSON but schema validation failed for cell {cell_pos}")
                    print(f"  DEBUG: parsed JSON = {json.dumps(result, ensure_ascii=False)}")
                    
                    self.error_tracker.record("json_extraction_errors")
                    retry_count += 1
                    last_error_message = f"Invalid JSON schema. Response: {output_text[:200]}..."
                
                if attempt < self.config.max_retries - 1:
                    print(f"  ⚠️  Cell {cell_pos}: JSON extraction failed "
                          f"(attempt {attempt + 1}/{self.config.max_retries}), retrying...")
                    time.sleep(self.config.retry_delay)
                    
            except Exception as e:
                # API error
                self.error_tracker.record("api_errors")
                retry_count += 1
                last_error_message = str(e)
                
                if attempt < self.config.max_retries - 1:
                    print(f"  ⚠️  Cell {cell_pos}: API error "
                          f"(attempt {attempt + 1}/{self.config.max_retries}): {e}")
                    print(f"     Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)
        
        # All retries exhausted - raise exception to halt simulation
        # This ensures data integrity by not continuing with partially updated grids
        raise LOGOSSimulationError(
            cell_pos,
            f"Failed after {self.config.max_retries} attempts. Last error: {last_error_message}"
        )
    
    def _update_grid_parallel(self, grid: np.ndarray, current_step: int) -> np.ndarray:
        """
        Update all cells in the grid using parallel LLM queries.
        
        This method distributes cell update tasks across multiple threads
        for efficient parallel processing.
        
        Args:
            grid: Current grid state as a 2D numpy array.
            current_step: Current simulation step number.
        
        Returns:
            New grid state after all cells have been updated.
        
        Raises:
            LOGOSSimulationError: If any cell update fails after all retries.
        """
        size_y, size_x = grid.shape
        new_grid = np.empty((size_y, size_x), dtype=object)
        
        # Prepare tasks for all cells
        tasks = []
        neighborhood_func = (
            get_sparse_region_von_neumann_neighborhood
            if size_x == 10 and size_y == 10
            else get_moore_neighborhood
        )
        for i in range(size_y):
            for j in range(size_x):
                target_cell = grid[i, j]
                neighbors = neighborhood_func(grid, i, j)
                tasks.append((target_cell, neighbors, (i, j)))
        
        print(f"  Processing {len(tasks)} cells in parallel "
              f"(max {self.config.max_workers} workers)...")
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_cell = {
                executor.submit(
                    self._query_llm_for_next_state,
                    task[0], task[1], task[2], current_step
                ): task[2]
                for task in tasks
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_cell):
                cell_pos = future_to_cell[future]
                
                # This will raise LOGOSSimulationError if the cell update failed
                # The exception propagates up to halt the simulation
                (i, j), next_state = future.result()
                new_grid[i, j] = next_state
                
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"  [{completed}/{len(tasks)}] Completed")
        
        return new_grid
    

    def _query_llm_for_seed_state(
        self,
        neighbors: List[Dict[str, str]],
        cell_pos: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], str]:
        """
        Query the LLM to determine the next seed state of a cell.

        This method uses the early-phase prompt configuration and does not
        provide the current target cell as part of the user prompt.

        Args:
            neighbors: List of neighbor descriptions from get_von_neumann_neighborhood().
            cell_pos: (row, col) tuple identifying the cell position.

        Returns:
            Tuple of (cell_pos, next_state_description).

        Raises:
            LOGOSSimulationError: If the cell update fails after all retries.
        """
        normalized_neighbors = [
            {
                "position": n["position"],
                "description": self._normalize_cell_state_text(n["description"]),
            }
            for n in neighbors
        ]

        neighbor_desc = "\n".join([
            f"- {n['position']}: {n['description']}"
            for n in normalized_neighbors
        ])

        user_prompt = self.EARLY_PHASE_USER_PROMPT_TEMPLATE.format(
            neighbor_desc=neighbor_desc
        )

        # Track this cell
        self.error_tracker.record("total_cells_processed")
        retry_count = 0
        last_error_message = ""

        # Retry loop
        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting delay
                time.sleep(self.config.delay_between_calls)

                response = self._client.responses.create(
                    model=self.config.model_name,
                    instructions=self.EARLY_PHASE_SYSTEM_INSTRUCTIONS,
                    input=user_prompt,
                    temperature=self.config.temperature,
                )

                if response.usage:
                    self.usage_tracker.record({
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.total_tokens,
                    })

                output_text = response.output_text
                result = extract_json_from_response(output_text)

                if result is None:
                    print(f"\n===== RAW LLM RESPONSE for cell {cell_pos}, attempt {attempt + 1} =====")
                    print(output_text)
                    print("===== END RAW LLM RESPONSE =====\n")
                    print(f"  DEBUG: extract_json_from_response returned None for cell {cell_pos}")

                    self.error_tracker.record("json_extraction_errors")
                    retry_count += 1
                    last_error_message = f"JSON extraction failed. Response: {output_text[:200]}..."
                else:
                    normalized_result, sketch_was_invalid = self._normalize_report_json(result)

                    if sketch_was_invalid:
                        self.error_tracker.record("sketch_rgb_16x16_failures")

                    if normalized_result and self._is_valid_report_json(normalized_result):
                        if sketch_was_invalid:
                            self.error_tracker.record("sketch_rgb_16x16_normalizations")

                        if attempt == 0:
                            self.error_tracker.record("successful_first_try")
                        else:
                            self.error_tracker.record("successful_after_retry")
                        return cell_pos, json.dumps(normalized_result, ensure_ascii=False)

                    print(f"\n===== RAW LLM RESPONSE for cell {cell_pos}, attempt {attempt + 1} =====")
                    print(output_text)
                    print("===== END RAW LLM RESPONSE =====\n")
                    print(f"  DEBUG: extracted JSON but schema validation failed for cell {cell_pos}")
                    print(f"  DEBUG: parsed JSON = {json.dumps(result, ensure_ascii=False)}")

                    self.error_tracker.record("json_extraction_errors")
                    retry_count += 1
                    last_error_message = f"Invalid JSON schema. Response: {output_text[:200]}..."

                if attempt < self.config.max_retries - 1:
                    print(f"  ⚠️  Cell {cell_pos}: JSON extraction failed "
                          f"(attempt {attempt + 1}/{self.config.max_retries}), retrying...")
                    time.sleep(self.config.retry_delay)

            except Exception as e:
                self.error_tracker.record("api_errors")
                retry_count += 1
                last_error_message = str(e)

                if attempt < self.config.max_retries - 1:
                    print(f"  ⚠️  Cell {cell_pos}: API error "
                          f"(attempt {attempt + 1}/{self.config.max_retries}): {e}")
                    print(f"     Retrying in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)

        raise LOGOSSimulationError(
            cell_pos,
            f"Failed after {self.config.max_retries} attempts. Last error: {last_error_message}"
        )

    def _update_seed_grid_parallel(self, grid: np.ndarray) -> np.ndarray:
        """
        Update all cells in the 5x5 seed grid using parallel LLM queries.

        Args:
            grid: Current seed grid state as a 2D numpy array.

        Returns:
            New seed grid state after all cells have been updated.

        Raises:
            LOGOSSimulationError: If any cell update fails after all retries.
        """
        size_y, size_x = grid.shape
        new_grid = np.empty((size_y, size_x), dtype=object)

        tasks = []
        for i in range(size_y):
            for j in range(size_x):
                neighbors = get_von_neumann_neighborhood(grid, i, j)
                tasks.append((neighbors, (i, j)))

        print(f"  Processing {len(tasks)} seed cells in parallel "
              f"(max {self.config.max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_cell = {
                executor.submit(
                    self._query_llm_for_seed_state,
                    task[0], task[1]
                ): task[1]
                for task in tasks
            }

            completed = 0
            for future in as_completed(future_to_cell):
                cell_pos = future_to_cell[future]
                (i, j), next_state = future.result()
                new_grid[i, j] = next_state

                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"  [seed {completed}/{len(tasks)}] Completed")

        return new_grid

    def _build_region_seed_history(self) -> List[np.ndarray]:
        """
        Build the 5x5 seed history used to initialize the four regions.

        Returns:
            List of 5x5 grid states including the initial placeholder state
            and the generated seed evolution states.
        """
        print(f"\n🌱 Generating 5x5 seed history for region initialization...")
        initialization_output_json = self._get_initialization_output_json()
        seed_grid = self.initializer(self.SEED_GRID_SIZE_X, self.SEED_GRID_SIZE_Y)
        seed_history = [seed_grid.copy()]
        self._save_initialization_results(seed_history, initialization_output_json)

        seed_start_time = time.time()

        for seed_step in range(1, self.SEED_GENERATION_STEPS + 1):
            print(f"\n{'-' * 70}")
            print(f"🌱 Seed Step {seed_step}/{self.SEED_GENERATION_STEPS}")
            print(f"{'-' * 70}")

            seed_step_start = time.time()
            before_stats = self.usage_tracker.get_stats()

            seed_grid = self._update_seed_grid_parallel(seed_grid)
            seed_history.append(seed_grid.copy())

            self._save_initialization_results(seed_history, initialization_output_json)

            after_stats = self.usage_tracker.get_stats()
            delta = {
                "input_tokens": after_stats["input_tokens"] - before_stats["input_tokens"],
                "output_tokens": after_stats["output_tokens"] - before_stats["output_tokens"],
                "total_tokens": after_stats["total_tokens"] - before_stats["total_tokens"],
                "calls": after_stats["calls"] - before_stats["calls"],
            }

            step_tracker = UsageTracker()
            step_tracker._stats = delta
            step_cost, step_in_cost, step_out_cost = step_tracker.calculate_cost(self.config.model_name)

            if step_cost is not None:
                print(f"  💵 Seed step {seed_step} cost (est.): ${step_cost:.4f} "
                      f"(input ${step_in_cost:.4f}, output ${step_out_cost:.4f}) | "
                      f"calls {delta['calls']}, tokens in/out "
                      f"{delta['input_tokens']}/{delta['output_tokens']}")

            seed_step_time = time.time() - seed_step_start
            print(f"  ⏱️  Seed step completed in {seed_step_time:.2f} seconds")

            seed_elapsed = time.time() - seed_start_time
            avg_seed_time = seed_elapsed / seed_step if seed_step > 0 else 0
            seed_remaining = self.SEED_GENERATION_STEPS - seed_step
            seed_eta = avg_seed_time * seed_remaining

            print(f"  📊 Seed progress: {seed_step}/{self.SEED_GENERATION_STEPS} "
                  f"({seed_step/self.SEED_GENERATION_STEPS*100:.1f}%)")
            if seed_eta > 0:
                print(f"  ⏰ Seed ETA: {seed_eta/60:.1f} minutes")

        return seed_history

    def _compose_weakly_connected_initial_grid(self) -> np.ndarray:
        """
        Compose the 10x10 initial grid from four different 5x5 seed stages.

        Returns:
            A 10x10 grid initialized from four different stages of the
            generated 5x5 seed history.
        """
        seed_history = self._build_region_seed_history()
        grid = np.empty((self.config.grid_size_y, self.config.grid_size_x), dtype=object)

        region_sources = [
            ((0, 5), (0, 5), self.REGION_SOURCE_STEPS[0]),
            ((0, 5), (5, 10), self.REGION_SOURCE_STEPS[1]),
            ((5, 10), (0, 5), self.REGION_SOURCE_STEPS[2]),
            ((5, 10), (5, 10), self.REGION_SOURCE_STEPS[3]),
        ]

        for row_range, col_range, source_step in region_sources:
            source_grid = seed_history[source_step]
            for local_i, global_i in enumerate(range(row_range[0], row_range[1])):
                for local_j, global_j in enumerate(range(col_range[0], col_range[1])):
                    grid[global_i, global_j] = source_grid[local_i, local_j]

        return grid

    def _initialize_grid(self) -> np.ndarray:
        """
        Initialize the grid for a new simulation.

        For the 10x10 setup, this method generates a 5x5 seed history and uses
        different seed stages to initialize the four weakly coupled 5x5 regions.
        For all other grid sizes, it falls back to the user-provided initializer.

        Returns:
            Initialized simulation grid.
        """
        if self.config.grid_size_x == 10 and self.config.grid_size_y == 10:
            return self._compose_weakly_connected_initial_grid()
        return self.initializer(self.config.grid_size_x, self.config.grid_size_y)

    def _get_initialization_output_json(self) -> str:
        """
        Get the JSON filename used for saving initialization-stage results.

        Returns:
            Path to the initialization-stage output JSON file.
        """
        root, ext = os.path.splitext(self.config.output_json)
        if not ext:
            ext = ".json"
        return f"{root}_initialization{ext}"

    def _save_initialization_results(self, seed_history: List[np.ndarray], filename: str) -> None:
        """
        Save initialization-stage seed history to a JSON file.

        Args:
            seed_history: List of 5x5 seed grid states.
            filename: Path to the output JSON file.
        """
        result = {
            "metadata": {
                "grid_size_x": self.SEED_GRID_SIZE_X,
                "grid_size_y": self.SEED_GRID_SIZE_Y,
                "max_steps": self.SEED_GENERATION_STEPS,
                "model": self.config.model_name,
                "timestamp": datetime.now().isoformat(),
                "stage": "initialization",
            },
            "steps": []
        }

        for step_num, grid in enumerate(seed_history):
            step_data = {
                "step": step_num,
                "grid": grid_to_dict(grid)
            }
            result["steps"].append(step_data)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Initialization result saved to: {filename}")

    def _save_results(self, filename: str) -> None:
        """
        Save simulation results to a JSON file.
        
        Args:
            filename: Path to the output JSON file.
        """
        result = {
            "metadata": {
                "grid_size_x": self.config.grid_size_x,
                "grid_size_y": self.config.grid_size_y,
                "max_steps": self.config.max_steps,
                "model": self.config.model_name,
                "timestamp": datetime.now().isoformat(),
            },
            "steps": []
        }
        
        for step_num, grid in enumerate(self.history):
            step_data = {
                "step": step_num,
                "grid": grid_to_dict(grid)
            }
            result["steps"].append(step_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Simulation result saved to: {filename}")
    
    def _load_results(self, filename: str) -> Tuple[Optional[Dict], List[np.ndarray]]:
        """
        Load simulation results from a JSON file.
        
        Args:
            filename: Path to the JSON file to load.
        
        Returns:
            Tuple of (metadata dict, list of grid states).
            Returns (None, []) if loading fails.
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data['metadata']
            size_x = metadata['grid_size_x']
            size_y = metadata['grid_size_y']
            
            history = []
            for step_data in data['steps']:
                grid = dict_to_grid(step_data['grid'], size_x, size_y)
                history.append(grid)
            
            return metadata, history
        except FileNotFoundError:
            return None, []
        except Exception as e:
            print(f"⚠️  Warning: Failed to load existing simulation: {e}")
            return None, []
    
    def run(self) -> List[np.ndarray]:
        """
        Run the LOGOS-CA simulation.
        
        This is the main entry point for running a simulation. It handles:
        - API key validation
        - Resume from previous runs (if enabled)
        - Grid initialization
        - Step-by-step simulation with progress tracking
        - Cost estimation and error statistics
        
        If a cell update fails after exhausting all retry attempts, the
        simulation is halted and a LOGOSSimulationError is raised. The
        progress up to the failure point is saved.
        
        Returns:
            List of grid states (numpy arrays) for each simulation step.
        
        Raises:
            RuntimeError: If OpenAI API key is not configured.
            LOGOSSimulationError: If a cell update fails after all retries.
        """
        print("=" * 70)
        print("  LOGOS-CA: Language Oriented Grid Of Statements - Cellular Automaton")
        print("=" * 70)
        
        # Validate API client
        if not self._client:
            raise RuntimeError(
                "OpenAI API key is not set! "
                "Set OPENAI_API_KEY environment variable or pass api_key in config."
            )
        
        # Print configuration
        print(f"\n⚙️  Configuration:")
        print(f"   Grid size: {self.config.grid_size_x}x{self.config.grid_size_y}")
        print(f"   Max steps: {self.config.max_steps}")
        print(f"   Model: {self.config.model_name}")
        print(f"   Max parallel workers: {self.config.max_workers}")
        print(f"   LLM retry attempts: {self.config.max_retries}")
        print(f"   Output file: {self.config.output_json}")
        print(f"   Initialization output file: {self._get_initialization_output_json()}")
        print(f"   Resume mode: {'Enabled' if self.config.enable_resume else 'Disabled'}")
        
        # Show pricing info
        if self.config.model_name in MODEL_PRICES_USD_PER_1M:
            p = MODEL_PRICES_USD_PER_1M[self.config.model_name]
            print(f"   Price (est.): input ${p['input']}/1M tok, output ${p['output']}/1M tok")
        
        # Try to resume from existing results
        start_step = 0
        grid = None
        
        if self.config.enable_resume and os.path.exists(self.config.output_json):
            print(f"\n🔍 Found existing simulation file: {self.config.output_json}")
            metadata, self.history = self._load_results(self.config.output_json)
            
            if self.history:
                start_step = len(self.history) - 1
                grid = self.history[-1].copy()
                
                # Validate grid size
                if grid.shape != (self.config.grid_size_y, self.config.grid_size_x):
                    print(f"  ⚠️  Warning: Grid size mismatch!")
                    print(f"     Existing: {grid.shape[1]}x{grid.shape[0]}")
                    print(f"     Current config: {self.config.grid_size_x}x{self.config.grid_size_y}")
                    print(f"     Starting new simulation instead...")
                    self.history = []
                    start_step = 0
                    grid = None
                elif start_step >= self.config.max_steps:
                    print(f"  ℹ️  Simulation already completed (step {start_step}/{self.config.max_steps})")
                    return self.history
                else:
                    print(f"  ✓ Resuming from step {start_step}/{self.config.max_steps}")
                    print(f"     Remaining steps: {self.config.max_steps - start_step}")
        
        # Initialize new grid if needed
        if grid is None:
            print(f"\n🔄 Initializing new grid...")
            grid = self._initialize_grid()
            
            # Print sample of initial state
            print("\nInitial state (sample):")
            for i in range(min(3, self.config.grid_size_y)):
                for j in range(min(3, self.config.grid_size_x)):
                    desc = grid[i, j]
                    if len(desc) > 60:
                        desc = desc[:60] + "..."
                    print(f"  Cell ({i},{j}): {desc}")
            if self.config.grid_size_x > 3 or self.config.grid_size_y > 3:
                print("  ...")
            
            self.history = [grid.copy()]
            start_step = 0
        
        # Main simulation loop
        start_time = time.time()
        
        try:
            for step in range(start_step + 1, self.config.max_steps + 1):
                print(f"\n{'=' * 70}")
                print(f"🔄 Step {step}/{self.config.max_steps}")
                print(f"{'=' * 70}")
                
                step_start = time.time()
                
                # Snapshot usage before step
                before_stats = self.usage_tracker.get_stats()
                
                # Update grid (may raise LOGOSSimulationError)
                grid = self._update_grid_parallel(grid, step)
                self.history.append(grid.copy())
                
                # Save intermediate results
                self._save_results(self.config.output_json)
                
                # Calculate step costs
                after_stats = self.usage_tracker.get_stats()
                delta = {
                    "input_tokens": after_stats["input_tokens"] - before_stats["input_tokens"],
                    "output_tokens": after_stats["output_tokens"] - before_stats["output_tokens"],
                    "total_tokens": after_stats["total_tokens"] - before_stats["total_tokens"],
                    "calls": after_stats["calls"] - before_stats["calls"],
                }
                
                # Create temporary tracker for step cost calculation
                step_tracker = UsageTracker()
                step_tracker._stats = delta
                step_cost, step_in_cost, step_out_cost = step_tracker.calculate_cost(self.config.model_name)
                
                if step_cost is not None:
                    print(f"  💵 Step {step} cost (est.): ${step_cost:.4f} "
                          f"(input ${step_in_cost:.4f}, output ${step_out_cost:.4f}) | "
                          f"calls {delta['calls']}, tokens in/out "
                          f"{delta['input_tokens']}/{delta['output_tokens']}")
                
                step_time = time.time() - step_start
                print(f"  ⏱️  Step completed in {step_time:.2f} seconds")
                
                # Progress estimate
                elapsed = time.time() - start_time
                steps_completed = step - start_step
                avg_time = elapsed / steps_completed if steps_completed > 0 else 0
                remaining = self.config.max_steps - step
                eta = avg_time * remaining
                
                print(f"  📊 Progress: {step}/{self.config.max_steps} ({step/self.config.max_steps*100:.1f}%)")
                if eta > 0:
                    print(f"  ⏰ ETA: {eta/60:.1f} minutes")
        
        except LOGOSSimulationError as e:
            # Simulation halted due to cell update failure
            print(f"\n{'=' * 70}")
            print(f"❌ SIMULATION HALTED")
            print(f"{'=' * 70}")
            print(f"\nCell {e.cell_pos} failed to update after {self.config.max_retries} attempts.")
            print(f"Error: {e.message}")
            print(f"\nProgress has been saved to: {self.config.output_json}")
            print("You can investigate the issue and resume the simulation later.")
            
            # Print statistics collected so far
            self.error_tracker.print_summary()
            
            # Re-raise the exception
            raise
        
        # Final summary
        total_time = time.time() - start_time
        steps_run = self.config.max_steps - start_step
        
        print(f"\n{'=' * 70}")
        print(f"✅ Simulation completed!")
        print(f"   Total time: {total_time/60:.2f} minutes")
        if steps_run > 0:
            print(f"   Average time per step: {total_time/steps_run:.2f} seconds")
        print(f"{'=' * 70}")
        
        # Cost summary
        total_cost, in_cost, out_cost = self.usage_tracker.calculate_cost(self.config.model_name)
        if total_cost is not None:
            stats = self.usage_tracker.get_stats()
            print("\n🧾 Billing summary (estimated)")
            print(f"   Calls: {stats['calls']}")
            print(f"   Tokens: in {stats['input_tokens']} / out {stats['output_tokens']} "
                  f"(total {stats['total_tokens']})")
            print(f"   Cost:  ${total_cost:.4f} (input ${in_cost:.4f}, output ${out_cost:.4f})")
        
        # Error statistics
        self.error_tracker.print_summary()
        
        # Unique states count
        all_texts = set()
        for step_grid in self.history:
            all_texts.update(step_grid.flatten())
        
        print(f"\n📊 Statistics:")
        print(f"   Total unique states across all steps: {len(all_texts)}")
        print(f"   Total cells processed: "
              f"{self.config.grid_size_x * self.config.grid_size_y * steps_run}")
        print(f"\n💾 Results saved to: {self.config.output_json}")
        
        return self.history
