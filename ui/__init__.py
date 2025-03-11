"""
User interface package for ScholarLens
"""
from ui.main_page import render_main_layout
from ui.upload_view import render_upload_view
from ui.search_view import render_search_view
from ui.batch_view import render_batch_view
from ui.settings_view import render_settings_view

__all__ = [
    'render_main_layout',
    'render_upload_view',
    'render_search_view',
    'render_batch_view',
    'render_settings_view'
]