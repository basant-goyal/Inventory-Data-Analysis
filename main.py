import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read and prepare the data
def load_inventory_data():
    try:
        # Read CSV file
        df = pd.read_csv("Inventory-Records-Sample-Data.csv")
        
        # Remove any empty rows
        df = df.dropna(subset=['Product_ID'])
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['Opening_Stock', 'Purchase_Stock', 'Units_Sold', 'Hand_In_Stock', 'Cost_Per_Unit', 'Total_Cost']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except FileNotFoundError:
        print("Error: 'inventory_data.csv' file not found!")
        print("Please make sure the CSV file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

# Load data
df = load_inventory_data()

# Check if data was loaded successfully
if df is None:
    print("Failed to load data. Exiting...")
    exit()

# Display column names for verification
print("Column names in the dataset:")
print(df.columns.tolist())
print()

# Basic Data Analysis
print("=== INVENTORY DATA ANALYSIS ===")
print("\n1. Dataset Overview:")
print(f"Total Products: {len(df)}")
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\n2. Basic Statistics:")
print(df.describe())

# Calculate additional metrics using NumPy
df['Revenue'] = df['Units_Sold'] * df['Cost_Per_Unit']
df['Turnover_Rate'] = (df['Units_Sold'] / df['Opening_Stock']) * 100
df['Stock_Efficiency'] = (df['Hand_In_Stock'] / (df['Opening_Stock'] + df['Purchase_Stock'])) * 100

print("\n3. Key Performance Indicators:")
print(f"Total Revenue: ${df['Revenue'].sum():,.2f}")
print(f"Total Current Stock: {df['Hand_In_Stock'].sum()}")
print(f"Average Turnover Rate: {df['Turnover_Rate'].mean():.2f}%")
print(f"Products with Low Stock (<30): {len(df[df['Hand_In_Stock'] < 30])}")

# Top performing products
print("\n4. Top 5 Revenue Generators:")
top_revenue = df.nlargest(5, 'Revenue')[['Product_Name', 'Revenue']]
print(top_revenue)

print("\n5. Stock Level Analysis:")
low_stock = df[df['Hand_In_Stock'] < 30][['Product_Name', 'Hand_In_Stock']]
if len(low_stock) > 0:
    print("Products with Low Stock:")
    print(low_stock)
else:
    print("No products with critically low stock")

# Data Visualizations using Matplotlib and Seaborn
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Revenue by Product (Bar Chart)
top_products = df.nlargest(8, 'Revenue')
axes[0, 0].bar(range(len(top_products)), top_products['Revenue'])
axes[0, 0].set_title('Top Products by Revenue')
axes[0, 0].set_xlabel('Products')
axes[0, 0].set_ylabel('Revenue ($)')
axes[0, 0].set_xticks(range(len(top_products)))
axes[0, 0].set_xticklabels(top_products['Product_Name'], rotation=45, ha='right')

# 2. Stock Levels Comparison (Bar Chart)
stock_data = df[['Opening_Stock', 'Hand_In_Stock', 'Units_Sold']].head(8)
x = np.arange(len(stock_data))
width = 0.25

axes[0, 1].bar(x - width, stock_data['Opening_Stock'], width, label='Opening Stock')
axes[0, 1].bar(x, stock_data['Hand_In_Stock'], width, label='Current Stock')
axes[0, 1].bar(x + width, stock_data['Units_Sold'], width, label='Units Sold')
axes[0, 1].set_title('Stock Levels Comparison')
axes[0, 1].set_xlabel('Products')
axes[0, 1].set_ylabel('Quantity')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(df['Product_Name'].head(8), rotation=45, ha='right')
axes[0, 1].legend()

# 3. Turnover Rate Analysis (Line Chart)
axes[1, 0].plot(df['Product_Name'].head(10), df['Turnover_Rate'].head(10), marker='o')
axes[1, 0].set_title('Turnover Rate by Product')
axes[1, 0].set_xlabel('Products')
axes[1, 0].set_ylabel('Turnover Rate (%)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Cost vs Revenue Scatter Plot
axes[1, 1].scatter(df['Total_Cost'], df['Revenue'], alpha=0.7)
axes[1, 1].set_title('Cost vs Revenue Analysis')
axes[1, 1].set_xlabel('Total Cost ($)')
axes[1, 1].set_ylabel('Revenue ($)')

# Add product labels to scatter plot
for i, txt in enumerate(df['Product_Name']):
    if df['Revenue'].iloc[i] > 10000:  # Only label high revenue products
        axes[1, 1].annotate(txt[:8], (df['Total_Cost'].iloc[i], df['Revenue'].iloc[i]), 
                           fontsize=8, ha='center')

plt.tight_layout()
plt.show()

# Additional Analysis using Seaborn
plt.figure(figsize=(12, 8))

# Correlation Heatmap
plt.subplot(2, 2, 1)
correlation_data = df[['Opening_Stock', 'Units_Sold', 'Hand_In_Stock', 'Revenue', 'Turnover_Rate']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')

# Distribution of Stock Efficiency
plt.subplot(2, 2, 2)
sns.histplot(df['Stock_Efficiency'], bins=10, kde=True)
plt.title('Distribution of Stock Efficiency')
plt.xlabel('Stock Efficiency (%)')

# Box plot of Revenue by Product Category (simplified)
plt.subplot(2, 2, 3)
# Create simple categories based on cost
df['Category'] = pd.cut(df['Cost_Per_Unit'], bins=3, labels=['Low-Cost', 'Mid-Cost', 'High-Cost'])
sns.boxplot(data=df, x='Category', y='Revenue')
plt.title('Revenue Distribution by Cost Category')

# Sales Performance Trend
plt.subplot(2, 2, 4)
sns.barplot(data=df.head(8), x='Product_Name', y='Units_Sold')
plt.title('Sales Performance')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Summary Report
print("\n=== BUSINESS INTELLIGENCE INSIGHTS ===")
print("\n6. Key Findings:")
print(f"• Highest Revenue Product: {df.loc[df['Revenue'].idxmax(), 'Product_Name']} (${df['Revenue'].max():,.2f})")
print(f"• Best Turnover Rate: {df.loc[df['Turnover_Rate'].idxmax(), 'Product_Name']} ({df['Turnover_Rate'].max():.1f}%)")
print(f"• Most Efficient Stock Management: {df.loc[df['Stock_Efficiency'].idxmax(), 'Product_Name']} ({df['Stock_Efficiency'].max():.1f}%)")

print("\n7. Recommendations:")
high_revenue_low_turnover = df[(df['Revenue'] > df['Revenue'].mean()) & (df['Turnover_Rate'] < df['Turnover_Rate'].mean())]
if len(high_revenue_low_turnover) > 0:
    print("• Consider increasing marketing for high-value, slow-moving items:")
    for _, product in high_revenue_low_turnover.iterrows():
        print(f"  - {product['Product_Name']}")

print("• Monitor low stock items for reordering")
print("• Focus on high-turnover products for inventory expansion")

print("\n=== ANALYSIS COMPLETE ===")