import torch

from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict

log = get_pylogger(__name__)


def render(
    td: TensorDict, actions=None, ax=None, scale_xy: bool = True, vehicle_capacity=None,  return_ax: bool = False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib import cm, colormaps
    lw=1.5
    #num_routine = (actions == 0).sum().item() + 2
    #base = colormaps["nipy_spectral"]
    #color_list = base(np.linspace(0, 1, num_routine))
    #cmap_name = base.name + str(num_routine)
    #out = base.from_list(cmap_name, color_list, num_routine)

    if ax is None:
        _, ax = plt.subplots(dpi=100, figsize=(6, 6))

    td = td.detach().cpu()
    actions = actions.detach().cpu()
    num_depots = int(td["num_depots"].item())
    if actions is None:
        actions = td.get("action", None)

    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"]
    scale_demand = td["capacity_original"]
    demands_linehaul = td["demand_linehaul"] * scale_demand
    demands_backhaul = td["demand_backhaul"] * scale_demand
    num_depots = td["num_depots"]

    route_starts = ((actions[:-1] < num_depots)& (actions[1:] >= num_depots))

    num_routes = max(route_starts.sum().item(), 1,)

    base = colormaps["nipy_spectral"]
    color_list = [base(value) for value in np.linspace(0.05, 0.95, num_routes,)]


    # scale to closest integer
    if demands_linehaul.max() <= 1:  # fallback for no scaling
        # scale min value except 0 to 1 and max value to 9
        demands_linehaul = (
            (demands_linehaul - demands_linehaul.min())
            / (demands_linehaul.max() - demands_linehaul.min())
            * 9
        )
        demands_backhaul = (
            (demands_backhaul - demands_backhaul.min())
            / (demands_backhaul.max() - demands_backhaul.min())
            * 9
        )

        demands_linehaul = demands_linehaul.round().int()
        demands_backhaul = demands_backhaul.round().int()

    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        #actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])


        selected_depots = actions[actions < num_depots]

        last_depot = selected_depots[-1]

        plot_actions = torch.cat([actions, last_depot.reshape(1),])



    # Depot
    for idx in range(num_depots):
        ax.scatter(
            locs[idx, 0],
            locs[idx, 1],
            edgecolors=cm.Set2(2),
            facecolors="none",
            s=100,
            linewidths=1,
            marker="*",
            alpha=1,
        )

    for node_idx, loc in enumerate(locs):
        if node_idx < num_depots:
            continue  # skip depots
        delivery, pickup = demands_linehaul[node_idx], demands_backhaul[node_idx]
        if delivery > 0:
            ax.text(
                loc[0],
                loc[1] + 0.02,
                f"{delivery.item()}",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=10,
                color=cm.Set2(0),
            )
            # scatter delivery as downward triangle
            ax.scatter(
                loc[0],
                loc[1],
                edgecolors=cm.Set2(0),
                facecolors="none",
                s=30,
                linewidths=1,
                marker="v",
                alpha=1,
            )
        elif pickup > 0:
            ax.text(
                loc[0],
                loc[1] - 0.02,
                f"{pickup.item()}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=10,
                color=cm.Set2(1),
            )
            ax.scatter(
                loc[0],
                loc[1],
                edgecolors=cm.Set2(1),
                facecolors="none",
                s=30,
                linewidths=1,
                marker="^",
                alpha=1,
            )
        else:
            print("Error: no demand")
    # ---> Adapting to multi fleet
    selected_nodes = actions[actions < num_depots]
    current_depot = actions[0]
    last_depot = selected_nodes[-1]
    color_idx = 0
    vehicle_idx = -1
    plot_actions = torch.cat([actions, last_depot.reshape(1),])

    assert actions[0] < num_depots, "First action should be a depot"
    current_depot = actions[0]
    for ai, aj in zip(plot_actions[:-1], plot_actions[1:],):

        ai_is_depot = ai < num_depots
        aj_is_depot = aj < num_depots

        if ai_is_depot and aj_is_depot:
            current_depot = aj
            continue
        if ai_is_depot and not aj_is_depot:
            current_depot = ai
            vehicle_idx += 1
        if td["open_route"].item() and aj_is_depot:
            continue
        from_loc = locs[ai]
        if aj_is_depot:
            to_loc = locs[current_depot]
        else:
            to_loc = locs[aj]
        route_color = color_list[vehicle_idx % num_routes]

        if ai_is_depot or aj_is_depot:
            line_style = "--"
            alpha = 0.8
        else:
            line_style = "-"
            alpha = 1.0

        lw = 1.5

        ax.plot(
            [from_loc[0], to_loc[0]],
            [from_loc[1], to_loc[1]],
            color=route_color,
            lw=lw,
            alpha=alpha,
            linestyle=line_style,
        )

        ax.annotate(
            "",
            xy=(to_loc[0], to_loc[1]),
            xytext=(from_loc[0], from_loc[1]),
            arrowprops=dict(
                arrowstyle="->",
                color=route_color,
                lw=lw,
                alpha=alpha,
            ),
            size=15,
            annotation_clip=False,
        )

        # <---

    if scale_xy:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    if return_ax:
        return ax
    plt.show()



def render_animation(
    td,
    actions,
    filename="route_construction.gif",
    interval=500,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from matplotlib import colormaps
    from matplotlib.animation import FuncAnimation, PillowWriter

    td = td.detach().cpu()
    actions = actions.detach().cpu()

    if td.batch_size != torch.Size([]):
        td = td[0]

    if actions.ndim > 1:
        actions = actions[0]

    locs = td["locs"]
    num_depots = int(td["num_depots"].item())
    open_route = bool(td["open_route"].item())

    if actions.numel() == 0:
        raise ValueError("No actions were provided.")

    depot_indices = torch.arange(num_depots)

    if not torch.isin(actions[0], depot_indices):
        raise ValueError(
            "The first multi-depot action must select a depot."
        )

    # Convert the action sequence into physically travelled edges.
    segments = []

    current_depot = int(actions[0].item())
    current_node = current_depot
    vehicle_index = 0

    for next_action in actions[1:]:
        next_node = int(next_action.item())

        current_is_depot = current_node < num_depots
        next_is_depot = next_node < num_depots

        if next_is_depot:
            # A customer-to-depot action physically returns to the
            # depot from which the current vehicle departed.
            if not current_is_depot:
                if not open_route:
                    segments.append(
                        (
                            current_node,
                            current_depot,
                            vehicle_index,
                        )
                    )

                vehicle_index += 1

            # Selecting another depot does not represent travel.
            current_depot = next_node
            current_node = next_node
            continue

        # Travel from the current depot/customer to a customer.
        segments.append(
            (
                current_node,
                next_node,
                vehicle_index,
            )
        )

        current_node = next_node

    # RouteFinder normally stops after visiting the final customer,
    # so add its final depot return for a closed route.
    if current_node >= num_depots and not open_route:
        segments.append(
            (
                current_node,
                current_depot,
                vehicle_index,
            )
        )

    number_of_vehicles = max(
        (segment[2] for segment in segments),
        default=0,
    ) + 1

    colour_map = colormaps["turbo"]

    vehicle_colours = [
        colour_map(value)
        for value in np.linspace(
            0.05,
            0.95,
            number_of_vehicles,
        )
    ]

    fig, ax = plt.subplots(
        dpi=100,
        figsize=(7, 7),
    )

    def draw_nodes():
        # Depots
        ax.scatter(
            locs[:num_depots, 0],
            locs[:num_depots, 1],
            marker="s",
            s=120,
            facecolors="none",
            edgecolors="tab:green",
            linewidths=2,
            zorder=4,
        )

        for depot_index in range(num_depots):
            ax.text(
                locs[depot_index, 0],
                locs[depot_index, 1],
                f"D{depot_index}",
                ha="center",
                va="center",
                fontsize=8,
                zorder=5,
            )

        # Customers
        ax.scatter(
            locs[num_depots:, 0],
            locs[num_depots:, 1],
            marker="o",
            s=35,
            facecolors="none",
            edgecolors="tab:blue",
            zorder=3,
        )

    def update(frame):
        ax.clear()
        draw_nodes()

        for segment_index in range(frame):
            from_node, to_node, used_vehicle = segments[
                segment_index
            ]

            from_location = locs[from_node]
            to_location = locs[to_node]
            colour = vehicle_colours[used_vehicle]

            ax.annotate(
                "",
                xy=(
                    to_location[0],
                    to_location[1],
                ),
                xytext=(
                    from_location[0],
                    from_location[1],
                ),
                arrowprops={
                    "arrowstyle": "->",
                    "color": colour,
                    "lw": 1.5,
                    "alpha": 0.9,
                },
                annotation_clip=False,
            )

        if frame > 0:
            _, active_node, active_vehicle = segments[
                frame - 1
            ]

            ax.scatter(
                locs[active_node, 0],
                locs[active_node, 1],
                s=90,
                color=vehicle_colours[active_vehicle],
                zorder=6,
            )

            ax.set_title(
                f"Vehicle {active_vehicle + 1} — "
                f"step {frame}/{len(segments)}"
            )
        else:
            ax.set_title("Initial state")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    animation = FuncAnimation(
        fig,
        update,
        frames=len(segments) + 1,
        interval=interval,
        repeat=False,
    )

    frames_per_second = max(
        1,
        round(1000 / interval),
    )

    animation.save(
        filename,
        writer=PillowWriter(
            fps=frames_per_second,
        ),
    )

    plt.close(fig)

    return animation