"""
Chart styling module for consistent, elegant Plotly charts.

Use `register_plotly_template()` once at app startup to set the global
default template. Every subsequent `go.Figure()` will inherit these
settings automatically — no need to pass template= or plot_bgcolor=
in each update_layout call.
"""

import plotly.graph_objects as go
import plotly.io as pio

# Elegant color palette
COLORS = {
    'primary': '#1E3A5F',      # Navy blue
    'secondary': '#E07B53',    # Soft coral
    'accent': '#2E8B57',       # Sea green
    'positive': '#3498DB',     # Sky blue
    'negative': '#E74C3C',     # Soft red
    'neutral': '#555555',      # Dark gray
    'background': '#FFFFFF',
    'grid': '#E0E0E0',
    'text': '#1A1A1A',
    'text_light': '#333333',
    'uncertainty': 'rgba(230,126,34,0.25)',  # orange band for ±σ
}

# Chart template
PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': COLORS['background'],
        'plot_bgcolor': COLORS['background'],
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'size': 12,
            'color': COLORS['text'],
        },
        'title': {
            'font': {
                'size': 16,
                'color': COLORS['text'],
            },
            'x': 0.5,
            'xanchor': 'center',
        },
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'linecolor': COLORS['grid'],
            'linewidth': 1,
            'tickfont': {'size': 11},
            'title': {'font': {'size': 12}},
            'zeroline': False,
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'linecolor': COLORS['grid'],
            'linewidth': 1,
            'tickfont': {'size': 11},
            'title': {'font': {'size': 12}},
            'zeroline': False,
        },
        'legend': {
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': COLORS['grid'],
            'borderwidth': 1,
            'font': {'size': 11},
        },
        'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
        'hoverlabel': {
            'bgcolor': COLORS['background'],
            'bordercolor': COLORS['grid'],
            'font': {'size': 12, 'color': COLORS['text']},
        },
    }
}


def get_elegant_layout(**kwargs):
    """
    Get an elegant layout dictionary for Plotly charts.
    
    Args:
        **kwargs: Additional layout parameters to override defaults
        
    Returns:
        dict: Layout configuration
    """
    layout = {
        'paper_bgcolor': COLORS['background'],
        'plot_bgcolor': COLORS['background'],
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'size': 12,
            'color': COLORS['text'],
        },
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'linecolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'linecolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
        },
        'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
        'hoverlabel': {
            'bgcolor': COLORS['background'],
            'font': {'size': 12},
        },
        'legend': {
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': COLORS['grid'],
            'borderwidth': 1,
        },
    }
    
    # Override with kwargs
    for key, value in kwargs.items():
        if isinstance(value, dict) and key in layout and isinstance(layout[key], dict):
            layout[key].update(value)
        else:
            layout[key] = value
    
    return layout


def style_timeseries_line():
    """Get styling for time series line."""
    return dict(color=COLORS['primary'], width=2)


def style_secondary_line():
    """Get styling for secondary line."""
    return dict(color=COLORS['secondary'], width=2)


def style_bar_colors(values):
    """
    Get colors for bar chart based on positive/negative values.
    
    Args:
        values: Array of values
        
    Returns:
        list: Colors for each bar
    """
    return [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in values]


def get_hline_style(color='gray', dash='dash'):
    """Get styling for horizontal lines."""
    return dict(
        line_color=color,
        line_width=1,
        line_dash=dash,
    )


# ---------------------------------------------------------------------------
# Global Plotly template — call once at app startup
# ---------------------------------------------------------------------------

_TEMPLATE_NAME = 'arcfresh_elegant'


def register_plotly_template():
    """Register and activate the ARCFRESH elegant template as Plotly default.

    Call this **once** at Streamlit app startup (e.g. in app/main.py).
    After registration every ``go.Figure()`` automatically inherits:
    - white background with correct dark-text colors
    - Inter font family
    - subtle grid lines (#E8E8E8)
    - proper hover-label styling

    Charts that already specify ``template="plotly_white"`` or explicit
    ``plot_bgcolor='white'`` will still pick up the font / color settings
    because the default template is merged first.
    """
    _FONT = dict(
        family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
        size=12,
        color=COLORS['text'],
    )

    _layout = go.Layout(
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=_FONT,
        title=dict(
            font=dict(size=16, color=COLORS['text']),
            x=0.5,
            xanchor='center',
        ),
        xaxis=dict(
            gridcolor=COLORS['grid'],
            gridwidth=1,
            linecolor=COLORS['grid'],
            linewidth=1,
            tickfont=dict(size=11, color=COLORS['text']),
            title=dict(font=dict(size=12, color=COLORS['text'])),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            gridwidth=1,
            linecolor=COLORS['grid'],
            linewidth=1,
            tickfont=dict(size=11, color=COLORS['text']),
            title=dict(font=dict(size=12, color=COLORS['text'])),
            zeroline=False,
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=COLORS['grid'],
            borderwidth=1,
            font=dict(size=11, color=COLORS['text']),
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor=COLORS['background'],
            bordercolor=COLORS['grid'],
            font=dict(size=12, color=COLORS['text']),
        ),
        hovermode='x unified',
        colorway=[
            COLORS['primary'], COLORS['secondary'], COLORS['accent'],
            COLORS['positive'], COLORS['negative'], '#9B59B6',
            '#F39C12', '#1ABC9C', '#34495E',
        ],
    )

    template = go.layout.Template(layout=_layout)
    pio.templates[_TEMPLATE_NAME] = template
    # Set as default (merged with plotly base so scatter/bar defaults still work)
    pio.templates.default = f'plotly+{_TEMPLATE_NAME}'


def apply_elegant_style(fig: go.Figure) -> go.Figure:
    """Apply the elegant style to an existing figure (post-hoc).

    Useful when a figure was built by third-party code that does not
    respect the global template.  Modifies the figure **in-place** and
    returns it for chaining convenience.
    """
    fig.update_layout(
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=12,
            color=COLORS['text'],
        ),
        xaxis=dict(
            gridcolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text']),
            title_font=dict(color=COLORS['text']),
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text']),
            title_font=dict(color=COLORS['text']),
        ),
        legend=dict(font=dict(color=COLORS['text'])),
        hoverlabel=dict(font=dict(color=COLORS['text'])),
    )
    return fig
