"""
Exploratory Data Analysis module for complaint data.

This module provides functions to analyze and visualize the complaint dataset.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EDAAnalysis:
    """Class for performing exploratory data analysis on complaint data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA analysis with a dataframe.
        
        Args:
            df: Dataframe containing complaint data.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty")
        
        self.df = df.copy()
        logger.info(f"Initialized EDA analysis with {len(self.df)} rows")
    
    def get_basic_stats(self) -> dict:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dictionary with basic statistics.
        """
        stats = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        logger.info("Basic statistics calculated")
        return stats
    
    def analyze_product_distribution(self) -> pd.DataFrame:
        """
        Analyze distribution of complaints across products.
        
        Returns:
            DataFrame with product counts and percentages.
        """
        if 'Product' not in self.df.columns:
            raise ValueError("'Product' column not found in dataframe")
        
        product_dist = self.df['Product'].value_counts().reset_index()
        product_dist.columns = ['Product', 'Count']
        product_dist['Percentage'] = (
            product_dist['Count'] / len(self.df) * 100
        ).round(2)
        
        logger.info("Product distribution calculated")
        return product_dist
    
    def analyze_narrative_length(self) -> pd.DataFrame:
        """
        Analyze the length (word count) of complaint narratives.
        
        Returns:
            DataFrame with narrative length statistics.
        """
        if 'Consumer complaint narrative' not in self.df.columns:
            raise ValueError("'Consumer complaint narrative' column not found")
        
        # Calculate word counts
        narratives = self.df['Consumer complaint narrative'].astype(str)
        word_counts = narratives.str.split().str.len()
        
        stats = {
            'mean': word_counts.mean(),
            'median': word_counts.median(),
            'std': word_counts.std(),
            'min': word_counts.min(),
            'max': word_counts.max(),
            'q25': word_counts.quantile(0.25),
            'q75': word_counts.quantile(0.75),
            'q90': word_counts.quantile(0.90),
            'q95': word_counts.quantile(0.95),
            'q99': word_counts.quantile(0.99)
        }
        
        logger.info("Narrative length statistics calculated")
        return pd.DataFrame([stats])
    
    def count_narratives(self) -> dict:
        """
        Count complaints with and without narratives.
        
        Returns:
            Dictionary with counts of narratives.
        """
        if 'Consumer complaint narrative' not in self.df.columns:
            raise ValueError("'Consumer complaint narrative' column not found")
        
        narratives = self.df['Consumer complaint narrative']
        
        counts = {
            'total_complaints': len(self.df),
            'with_narrative': narratives.notna().sum(),
            'without_narrative': narratives.isna().sum(),
            'empty_narrative': (
                narratives.notna() & 
                (narratives.astype(str).str.strip() == '')
            ).sum(),
            'non_empty_narrative': (
                narratives.notna() & 
                (narratives.astype(str).str.strip() != '')
            ).sum()
        }
        
        counts['with_narrative_pct'] = (
            counts['with_narrative'] / counts['total_complaints'] * 100
        ).round(2)
        
        counts['without_narrative_pct'] = (
            counts['without_narrative'] / counts['total_complaints'] * 100
        ).round(2)
        
        logger.info("Narrative counts calculated")
        return counts
    
    def plot_product_distribution(
        self,
        save_path: Optional[str] = None,
        top_n: Optional[int] = None
    ) -> None:
        """
        Plot distribution of complaints across products.
        
        Args:
            save_path: Path to save the plot. If None, displays plot.
            top_n: Number of top products to show. If None, shows all.
        """
        product_dist = self.analyze_product_distribution()
        
        if top_n:
            product_dist = product_dist.head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=product_dist, x='Product', y='Count')
        plt.title('Distribution of Complaints by Product', fontsize=16, fontweight='bold')
        plt.xlabel('Product', fontsize=12)
        plt.ylabel('Number of Complaints', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Product distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_narrative_length_distribution(
        self,
        save_path: Optional[str] = None,
        bins: int = 50
    ) -> None:
        """
        Plot distribution of narrative word counts.
        
        Args:
            save_path: Path to save the plot. If None, displays plot.
            bins: Number of bins for histogram.
        """
        if 'Consumer complaint narrative' not in self.df.columns:
            raise ValueError("'Consumer complaint narrative' column not found")
        
        narratives = self.df['Consumer complaint narrative'].astype(str)
        word_counts = narratives.str.split().str.len()
        
        plt.figure(figsize=(12, 6))
        plt.hist(word_counts, bins=bins, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Narrative Word Counts', fontsize=16, fontweight='bold')
        plt.xlabel('Word Count', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(word_counts.mean(), color='r', linestyle='--', 
                   label=f'Mean: {word_counts.mean():.1f}')
        plt.axvline(word_counts.median(), color='g', linestyle='--', 
                   label=f'Median: {word_counts.median():.1f}')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Narrative length distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report of the EDA findings.
        
        Returns:
            String containing the summary report.
        """
        basic_stats = self.get_basic_stats()
        product_dist = self.analyze_product_distribution()
        narrative_length = self.analyze_narrative_length()
        narrative_counts = self.count_narratives()
        
        report = f"""
EXPLORATORY DATA ANALYSIS SUMMARY
==================================

DATASET OVERVIEW
----------------
Total Rows: {basic_stats['total_rows']:,}
Total Columns: {basic_stats['total_columns']}
Memory Usage: {basic_stats['memory_usage_mb']:.2f} MB
Duplicate Rows: {basic_stats['duplicate_rows']:,}

PRODUCT DISTRIBUTION
--------------------
{product_dist.to_string(index=False)}

NARRATIVE STATISTICS
--------------------
Total Complaints: {narrative_counts['total_complaints']:,}
With Narrative: {narrative_counts['with_narrative']:,} ({narrative_counts['with_narrative_pct']}%)
Without Narrative: {narrative_counts['without_narrative']:,} ({narrative_counts['without_narrative_pct']}%)
Non-Empty Narrative: {narrative_counts['non_empty_narrative']:,}

NARRATIVE LENGTH (Word Count)
-----------------------------
Mean: {narrative_length['mean'].iloc[0]:.1f} words
Median: {narrative_length['median'].iloc[0]:.1f} words
Standard Deviation: {narrative_length['std'].iloc[0]:.1f} words
Minimum: {narrative_length['min'].iloc[0]:.0f} words
Maximum: {narrative_length['max'].iloc[0]:.0f} words
25th Percentile: {narrative_length['q25'].iloc[0]:.1f} words
75th Percentile: {narrative_length['q75'].iloc[0]:.1f} words
90th Percentile: {narrative_length['q90'].iloc[0]:.1f} words
95th Percentile: {narrative_length['q95'].iloc[0]:.1f} words
99th Percentile: {narrative_length['q99'].iloc[0]:.1f} words
"""
        
        logger.info("Summary report generated")
        return report

