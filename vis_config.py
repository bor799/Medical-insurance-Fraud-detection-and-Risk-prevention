VIS_CONFIG = {
    # Font settings
    'font': {
        'family': 'Times New Roman',
        'sizes': {
            'main_title': 30,      # Main title, bold, dark gray(#333)
            'axis_label': 30,      # Axis labels, 45 degree rotation
            'values': 25,          # Values in plots
            'sample_info': 25,     # Sample size info, light gray(#666)
            'annotation': 25       # All annotations
        },
        'colors': {
            'title': '#333333',    # Dark gray for titles
            'info': '#666666'      # Light gray for additional info
        }
    },
    
    # Figure dimensions (in cm)
    'figure': {
        'correlation': {
            'width': 10,
            'height': 10
        },
        'outliers': {
            'width': 24,
            'height': 10
        },
        'comparison': {
            'width': 12.5,         # Quarter A4 width
            'height': 10
        }
    },
    
    # Color schemes
    'colors': {
        'correlation': 'RdBu_r',   # Red-Blue diverging colormap
        'hexbin': 'Blues',         # Blues colormap for density plots
        'comparison': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'quaternary': '#d62728'
        }
    },
    
    # Plot elements
    'elements': {
        'scatter_size': 80,        # Size for scatter points
        'line_width': 3,          # Width for lines
        'grid': True,             # Show grid by default
        'dpi': 300                # High resolution for publication
    },
    
    # Special annotations
    'annotations': {
        'correlation': {
            'strong_correlation': 0.5,  # Threshold for white text
            'high_correlation': 0.8     # Threshold for gold border
        },
        'outliers': {
            'iqr_factor': 1.5,         # IQR factor for outlier detection
            'boundary_style': {
                'color': 'red',
                'linestyle': '--',
                'linewidth': 2
            }
        }
    },
    
    # Output settings
    'output': {
        'format': 'png',
        'dpi': 300,
        'bbox_inches': 'tight'
    }
}

# Plot style settings
PLOT_STYLE = {
    'figure.figsize': (10, 10),
    'font.family': 'Times New Roman',
    'font.size': 30,
    'font.weight': 'bold',
    'axes.titlesize': 30,
    'axes.labelsize': 30,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
    'axes.grid': True,
    'grid.alpha': 0.3
} 