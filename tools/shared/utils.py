"""Shared entities."""
from enum import Enum
from pathlib import Path
from typing import Type, Union, Callable, Optional

from bigym.bigym_env import BiGymEnv
from bigym.envs.cupboards_with_human import (
    HumanDrawerTopOpen,
    HumanDrawerTopClose,
    HumanDrawersAllOpen,
    HumanDrawersAllClose,
    HumanCupboardsCloseAll,
    HumanCupboardsOpenAll,
    HumanWallCupboardOpen,
    HumanWallCupboardClose,
)
from bigym.envs.cupboards import (
    DrawerTopOpen,
    DrawerTopClose,
    DrawersAllOpen,
    DrawersAllClose,
    CupboardsCloseAll,
    CupboardsOpenAll,
    WallCupboardOpen,
    WallCupboardClose,
)
from bigym.envs.dishwasher import (
    DishwasherOpen,
    DishwasherClose,
    DishwasherOpenTrays,
    DishwasherCloseTrays,
)
from bigym.envs.dishwasher_cups import (
    DishwasherUnloadCups,
    DishwasherLoadCups,
    DishwasherUnloadCupsLong,
)
from bigym.envs.dishwasher_cutlery import (
    DishwasherUnloadCutlery,
    DishwasherLoadCutlery,
    DishwasherUnloadCutleryLong,
)
from bigym.envs.dishwasher_plates import (
    DishwasherUnloadPlates,
    DishwasherLoadPlates,
    DishwasherUnloadPlatesLong,
)
from bigym.envs.groceries import GroceriesStoreLower, GroceriesStoreUpper
from bigym.envs.manipulation import FlipCup, FlipCutlery, StackBlocks
from bigym.envs.move_plates import MovePlate, MoveTwoPlates
from bigym.envs.pick_and_place import (
    TakeCups,
    PutCups,
    PickBox,
    StoreBox,
    SaucepanToHob,
    StoreKitchenware,
    ToastSandwich,
    FlipSandwich,
    RemoveSandwich,
)
from bigym.envs.reach_target import ReachTarget, ReachTargetDual, ReachTargetSingle
from bigym.robots.robot import Robot
from bigym.robots.configs.google_robot import GoogleRobot
from bigym.robots.configs.h1 import H1, H1FineManipulation
from bigym.robots.configs.stretch import StretchRobot
from demonstrations.const import SAFETENSORS_SUFFIX

from dearpygui import dearpygui as dpg

from vr.viewer.control_profiles.control_profile import ControlProfile
from vr.viewer.control_profiles.h1_floating import H1Floating
from vr.viewer.control_profiles.universal_floating import UniversalFloating


class ReplayMode(Enum):
    """Enum controlling joint position mode during demo replay."""

    Absolute = 0
    Delta = 1


REPLAY_MODES: dict[str, ReplayMode] = {
    "Absolute": ReplayMode.Absolute,
    "Delta": ReplayMode.Delta,
}

ENVIRONMENTS: dict[str, Type[BiGymEnv]] = {
    "Reach Target": ReachTarget,
    "Reach Target Single": ReachTargetSingle,
    "Reach Target Dual": ReachTargetDual,
    "Stack Blocks": StackBlocks,
    "Move Plate": MovePlate,
    "Move Two Plates": MoveTwoPlates,
    "Dishwasher Open": DishwasherOpen,
    "Dishwasher Close": DishwasherClose,
    "Dishwasher Open Trays": DishwasherOpenTrays,
    "Dishwasher Close Trays": DishwasherCloseTrays,
    "Dishwasher Unload Plates": DishwasherUnloadPlates,
    "Dishwasher Unload Plates Long": DishwasherUnloadPlatesLong,
    "Dishwasher Load Plates": DishwasherLoadPlates,
    "Dishwasher Unload Cutlery": DishwasherUnloadCutlery,
    "Dishwasher Unload Cutlery Long": DishwasherUnloadCutleryLong,
    "Dishwasher Load Cutlery": DishwasherLoadCutlery,
    "Dishwasher Unload Cups": DishwasherUnloadCups,
    "Dishwasher Unload Cups Long": DishwasherUnloadCupsLong,
    "Dishwasher Load Cups": DishwasherLoadCups,
    "Drawer Top Open": DrawerTopOpen,
    "Drawer Top Close": DrawerTopClose,
    "Drawers All Open": DrawersAllOpen,
    "Drawers All Close": DrawersAllClose,
    "Wall Cupboard Open": WallCupboardOpen,
    "Wall Cupboard  Close": WallCupboardClose,
    "Cupboards Open All": CupboardsOpenAll,
    "Cupboards Close All": CupboardsCloseAll,
    "Take Cups": TakeCups,
    "Put Cups": PutCups,
    "Flip Cup": FlipCup,
    "Flip Cutlery": FlipCutlery,
    "Pick Box": PickBox,
    "Store Box": StoreBox,
    "Saucepan To Hob": SaucepanToHob,
    "Store Kitchenware": StoreKitchenware,
    "Toast Sandwich": ToastSandwich,
    "Flip Sandwich": FlipSandwich,
    "Remove Sandwich": RemoveSandwich,
    "Groceries Store Lower": GroceriesStoreLower,
    "Groceries Store Upper": GroceriesStoreUpper,
    "Human Cupboards Drawer Top Open": HumanDrawerTopOpen,
    "Human Cupboards Drawer Top Close": HumanDrawerTopClose,
    "Human Cupboards Drawers All Open": HumanDrawersAllOpen,
    "Human Cupboards Drawers All Close": HumanDrawersAllClose,
    "Human Cupboards Wall Cupboard Open": HumanWallCupboardOpen,
    "Human Cupboards Wall Cupboard  Close": HumanWallCupboardClose,
    "Human Cupboards Cupboards Open All": HumanCupboardsOpenAll,
    "Human Cupboards Cupboards Close All": HumanCupboardsCloseAll,
}

ROBOTS: dict[str, Optional[Type[Robot]]] = {
    "Default": None,
    "H1": H1,
    "H1 Fine Manipulation": H1FineManipulation,
    "Google Robot": GoogleRobot,
    "Stretch Robot": StretchRobot,
}

CONTROL_PROFILES: dict[str, Type[ControlProfile]] = {
    "H1 Upper Body Floating": H1Floating,
    "Universal": UniversalFloating,
}


def get_demos_in_dir(directory: Path) -> list[Path]:
    """Get all demonstrations files in directory."""
    demos = list(directory.glob(f"*{SAFETENSORS_SUFFIX}"))
    return sorted(demos)


def select_directory(default_path: Union[Path, str], callback: Callable[[Path], None]):
    """Show directory selection dialog."""
    with dpg.file_dialog(
        modal=True,
        show=True,
        directory_selector=True,
        default_path=default_path,
        callback=lambda _, app_data: callback(Path(app_data["file_path_name"])),
        width=850,
        height=400,
    ):
        dpg.add_file_extension(".*")


def show_popup(
    header: str = "",
    message: str = "",
    actions: dict[str, Optional[Callable]] = None,
    loading_indicator: bool = False,
) -> int:
    """Show Popup."""

    def popup_callback(sender, app_data, user_data):
        popup_item = user_data["popup"]
        callback_action = user_data["action"]
        if callback_action:
            callback_action()
        dpg.delete_item(popup_item)

    def center():
        dpg.split_frame()
        window_width = dpg.get_viewport_width()
        window_height = dpg.get_viewport_height()
        popup_width, popup_height = dpg.get_item_rect_size(popup)
        x = (window_width - popup_width) / 2
        y = (window_height - popup_height) / 2
        dpg.set_item_pos(popup, [x, y])

    with dpg.window(
        label=header,
        modal=True,
        show=True,
        popup=True,
        no_resize=True,
        min_size=(400, 100),
        autosize=True,
        no_close=True,
    ) as popup:
        if message:
            dpg.add_text(message)
        if loading_indicator:
            dpg.add_loading_indicator(style=1)
        with dpg.group(horizontal=True):
            actions = actions or {}
            for label, action in actions.items():
                dpg.add_button(
                    label=label,
                    user_data={"popup": popup, "action": action},
                    callback=popup_callback,
                )
    center()
    return popup
