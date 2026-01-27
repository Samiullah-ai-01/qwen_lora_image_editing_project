"""
Dataset splitting utilities for SignForge.
"""

from __future__ import annotations
import random
from typing import List, Tuple, TypeVar, Any

from signforge.data.schema import DatasetItem

T = TypeVar('T')

def split_items(
    items: List[T], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    seed: int = 42
) -> Tuple[List[T], List[T], List[T]]:
    """
    Split a list into train, val, and test sets.
    
    Args:
        items: List of items to split
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed for reproducibility
        
    Returns:
        (train_list, val_list, test_list)
    """
    if not items:
        return [], [], []
        
    # Copy and shuffle
    shuffled = list(items)
    random.seed(seed)
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    
    return train, val, test


def stratify_by_domain(
    items: List[DatasetItem], 
    train_ratio: float = 0.8, 
    seed: int = 42
) -> Tuple[List[DatasetItem], List[DatasetItem]]:
    """
    Split items while ensuring domains are represented in both sets.
    """
    domain_groups: dict[str, List[DatasetItem]] = {}
    for item in items:
        domain = item.domain or "unknown"
        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append(item)
        
    train_total = []
    val_total = []
    
    for domain, domain_items in domain_groups.items():
        train, val, _ = split_items(domain_items, train_ratio, 1.0 - train_ratio, seed)
        train_total.extend(train)
        val_total.extend(val)
        
    return train_total, val_total
