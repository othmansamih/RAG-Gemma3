import gradio as gr
from typing import Tuple

class UISettings:
    """
    A class for handling UI-related settings and interactions.
    """
    
    @staticmethod
    def toggle_sidebar(state) -> Tuple[gr.update, bool]:
        """
        Toggles the visibility state of a sidebar.
        
        Args:
            state (bool): The current visibility state of the sidebar.
        
        Returns:
            Tuple[gr.update, bool]: An updated visibility state for Gradio components and the new state.
        """
        state = not state
        return gr.update(visible=state), state
