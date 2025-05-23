# Inventory Data Analysis

A Python-based data analysis project for tracking product stock levels, sales trends, and inventory optimization.

## Technologies Used

- **Python** - Core programming language
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization and plotting
- **Seaborn** - Statistical data visualization
- **Data Analytics** - Business intelligence and insights
- **Business Intelligence** - Performance metrics and KPIs

## Project Overview

This project performs exploratory data analysis on inventory dataset to:
- Track product stock levels and sales trends
- Develop comparative insights into stock fluctuations
- Analyze sales performance and cost efficiency
- Create data visualizations for decision-making
- Optimize inventory management strategies

## Features

### Data Analysis
- Dataset overview and basic statistics
- Revenue and turnover rate calculations
- Stock efficiency analysis
- Low stock identification and alerts

### Visualizations
- Revenue performance bar charts
- Stock level comparisons
- Turnover rate trend analysis
- Cost vs revenue scatter plots
- Correlation heatmaps
- Distribution analysis

### Business Intelligence
- Key Performance Indicators (KPIs)
- Top performing products identification
- Stock optimization recommendations
- Performance trend insights


## Data Structure

The analysis uses inventory data with the following columns:
- Product_ID: Unique product identifier
- Product_Name: Product description
- Opening_Stock: Initial inventory count
- Purchase_Stock: Additional stock purchased
- Units_Sold: Number of units sold
- Hand_In_Stock: Current inventory level
- Cost_Per_Unit: Unit cost in USD
- Total_Cost: Total inventory value

## Key Metrics

- **Total Revenue**: Sum of all product sales
- **Turnover Rate**: (Units Sold / Opening Stock) × 100
- **Stock Efficiency**: Current Stock / (Opening + Purchase Stock) × 100
- **Low Stock Alert**: Products with inventory < 30 units

## Output

The script generates:
1. Console output with key statistics and insights
2. Multiple visualization charts showing:
   - Top products by revenue
   - Stock level comparisons
   - Turnover rate trends
   - Cost vs revenue analysis
   - Correlation matrix
   - Performance distributions

## Business Insights

- Identifies top revenue generating products
- Highlights products with low stock levels
- Analyzes stock turnover efficiency
- Provides recommendations for inventory optimization
- Tracks sales performance trends

## Usage

Simply run the script to get comprehensive inventory analysis including statistical summaries, visualizations, and business recommendations for optimal inventory management.
