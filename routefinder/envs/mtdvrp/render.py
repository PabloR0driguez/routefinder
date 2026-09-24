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
