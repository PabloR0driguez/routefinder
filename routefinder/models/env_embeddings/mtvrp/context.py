import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index

from .echo_vehicle_encoder import EchoVehicleEncoder


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = step_context_dim if step_context_dim is not None else embed_dim
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class MTVRPContextEmbedding(EnvContext):
    """Context embedding MTVRP.
    - current time
    - used capacity
    - open route
    - remaining distance (set to default_remain_dist if positive infinity)

    Note that the distance limit (L) and open routes (O) are only embedding during decoding
    in this version
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(MTVRPContextEmbedding, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 4
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        mask = td["used_capacity_backhaul"] == 0
        used_capacity = torch.where(
            mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
        )
        available_load = td["vehicle_capacity"] - used_capacity
        remaining_dist = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"],
            posinf=self.default_remain_dist,
        )
        context_feats = torch.cat(
            (
                available_load,
                td["current_time"],
                td["open_route"].float(),
                remaining_dist,
            ),
            -1,
        )
        return context_feats


class RouteFinderContextEmbedding(EnvContext):
    """Context embedding MTVRP.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)

    We do not need to embed the open route here since it is done encoder-side.
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(RouteFinderContextEmbedding, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 3
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        mask = td["used_capacity_backhaul"] == 0
        used_capacity = torch.where(
            mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
        )
        available_load = td["vehicle_capacity"] - used_capacity
        remaining_dist = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"],
            posinf=self.default_remain_dist,
        )
        context_feats = torch.cat(
            (
                available_load,
                td["current_time"],
                remaining_dist,
            ),
            -1,
        )
        return context_feats


# Simple wrapper for better naming. We can replace with this name after getting
# the final models
class MTVRPContextEmbeddingRouteFinder(RouteFinderContextEmbedding):
    def __init__(self, *args, **kwargs):
        super(MTVRPContextEmbeddingRouteFinder, self).__init__(*args, **kwargs)


class MTVRPContextEmbeddingM(MTVRPContextEmbeddingRouteFinder):
    """Context embedding MTVRP with mixed backhaul.
    This is for the zero-shot or few-short on backhaul_class 2 instances.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 4)
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        context_feats = super(MTVRPContextEmbeddingM, self)._state_embedding(
            embeddings, td
        )
        # this will be 0 and tell the model we are *not* doing VRPMPD if backhaul class is not 2
        available_load_vrpmpd = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        ) * (td["backhaul_class"] == 2)

        # Note: now we need the projection to have embed_dim + 4 features!
        return torch.cat(
            (
                context_feats,
                available_load_vrpmpd,
            ),
            -1,
        )


class MTVRPContextEmbeddingFull(MTVRPContextEmbeddingRouteFinder):
    """Context embedding MTVRP with full features, including the Mixed Backhaul (MB) variants
    and Multi-depot (MD) variants as well.
    In practice, we use the same features as the MTVRPContextEmbeddingRouteFinder, but we add:
    - available load for MB variants (backhaul class 2)
    - locations of the depot where we started the route, since we may need to return there
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 4 + 2)
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        context_feats = super(MTVRPContextEmbeddingFull, self)._state_embedding(
            embeddings, td
        )
        # this will be 0 and tell the model we are *not* doing VRPMPD if backhaul class is not 2
        available_load_vrpmpd = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        ) * (td["backhaul_class"] == 2)

        start_depot_location = gather_by_index(td["locs"], td["current_depot"], dim=-2)

        return torch.cat(
            (
                context_feats,
                available_load_vrpmpd,
                start_depot_location,
            ),
            -1,
        )



#NEW CLASS FOR OUR FUTURE EMBEDDINGS, placeholder first

class MTVRPContextEmbeddingCurrentVehicle(MTVRPContextEmbeddingRouteFinder):
    def __init__(self, embed_dim=128,  default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 3 + 2)
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        #context_feats = super(MTVRPContextEmbeddingFull, self)._state_embedding(embeddings, td)
        existing_context = super()._state_embedding(embeddings, td,)


        vehicle_context = torch.cat((td["vehicle_capacity"],  td["speed"],  ),  dim=-1, )

        return torch.cat((existing_context, vehicle_context, ), dim=-1, )


#NEW CLASS FOR OUR FUTURE EMBEDDINGS, placeholder first
class MTVRPContextEmbeddingFleet(MTVRPContextEmbeddingRouteFinder):
    #inherits from MTVRPContextEmbeddingRouteFinder
    #inherit context features and methods
    def __init__(self, embed_dim=128, default_remain_dist=10):
        EnvContext.__init__(self, embed_dim=embed_dim, step_context_dim=embed_dim + 3+ embed_dim)
        self.default_remain_dist = default_remain_dist
        #embed_dim has current node embeddings, 
        #+3 is for availale capacity, current time, remaining distance, same as in RouteFinderContextEmbedding
        #+2 is for relative speed and relative capacity
        self.fleet_encoder = EchoVehicleEncoder(embed_dim=embed_dim)
        # EchoVehicleEncoder expects dims [Batch_num, Vehicle_num, Features_num]
        # Routefinder provides [Batch_num, Start_num, Vehicle_num, Features_num]
        # So we transform Routefinder's input into [Batch_numXStart_num, Vehicle_num, Features_num]
        # combining instances and starts into one dim
        # so vehicle mask must also match [Batch_numXStart_num, Vehicle_num, Features_num]
        # Node embeddings initially have [Batch_num, Node_num, Embedding_dim]
        # so we reshape them to [Batch_numXStart_num, Node_num, Embedding_dim],
        # so every attempt receives its corresponding node embeddings.
        # Current vehicle and current node each contain one index per attempt.
        # We flatten both into [Batch_numXStart_num],
        # so each combined batch entry has its corresponding indices.
        # The encoder returns [Batch_numXStart_num, Vehicle_num, Embedding_dim].
        # afterwards we restore [Batch_num, Start_num, Embedding_dim]
    def _state_embedding(self, embeddings, td):
        # context and fleet attributes
        #get available load, current time and remaining distance
        existing_context = super()._state_embedding(embeddings, td)
        #get speeds and capacities
        fleet_speeds = td["vehicle_speeds"]
        fleet_capacities = td["vehicle_capacities"]

        # Include the active vehicle and those still unused, up to the fleet budget.
        #we register vehicle index to later identify availability
        vehicle_indices = torch.arange(fleet_speeds.shape[-1], device=fleet_speeds.device)
        current_vehicle = td["current_vehicle"].reshape_as(td["speed"])
        fleet_size = td["num_vehicles"].reshape_as(current_vehicle)

        #include the active vehicle and unused vehicles within the budget.
        remaining = (vehicle_indices >= current_vehicle) & (vehicle_indices < fleet_size)
        remaining_speed = fleet_speeds.masked_fill(~remaining, 0).sum(dim=-1, keepdim=True)
        remaining_capacity = fleet_capacities.masked_fill(~remaining, 0).sum(dim=-1, keepdim=True)

        relative_speed = fleet_speeds / remaining_speed
        relative_capacity = fleet_capacities / remaining_capacity

        vehicle_features = torch.stack(
            (
                fleet_speeds,
                fleet_capacities,
                relative_speed,
                relative_capacity,
            ),
            dim=-1,
        )

        vehicle_features = vehicle_features.masked_fill(
            ~remaining.unsqueeze(-1), 0
        )

        # Encode the fleet and retrieve the active vehicle's context vector.
        fleet_context = self._fleet_embedding(
            vehicle_features, remaining, embeddings, td
        )

        # Append the learned vector to RouteFinder's three original features.
        return torch.cat((existing_context, fleet_context), dim=-1)

    def _fleet_embedding(self, vehicle_features, remaining, embeddings, td):
        batch_shape = vehicle_features.shape[:-2] #keeping dims: instance dim and start dim
        num_vehicles = vehicle_features.shape[-2] #keeping dims second to last: vehicles dim
        
        # vehicle_features last dim is vehicle attributes because of our torch.stack we did
        # in vehicle_generator.py we set speed = self.speed_sampler.sample((*batch_size, self.num_agents))
        # that is stored in td["vehicle_speeds"]
        # vehicle_features dims num of instances, num of starts, num vehicles and attributes per vehicle
        # embeddings dims: num of instances, nodes, embedding values
        # embeddings has no num of starts dim, so we reshape
        #
        if vehicle_features.ndim == 4 and embeddings.ndim == 3:
            embeddings = embeddings.unsqueeze(1).expand(
                -1, batch_shape[1], -1, -1
            )  #add a dim in position 1
        vehicle_features = vehicle_features.reshape(
            -1, num_vehicles, vehicle_features.shape[-1]
        )
        #we reshaped from [instances, starts, vehicles, embedding_values] to [batch, vehicles, embedding_values]

        remaining = remaining.reshape(-1, num_vehicles) #each row now contains the mask for the corresponding row of vehicle_features


        #embeddings.shape[-2] is the number of nodes
        # self.embed_dim is the embedding width
        # -1 infers the combined batch size
        node_embeddings = embeddings.reshape(-1, embeddings.shape[-2], self.embed_dim)
        visited = td["visited"].reshape(-1, td["visited"].shape[-1])
        vehicle_node_indices = torch.zeros_like(remaining, dtype=torch.long)
        current_vehicle = td["current_vehicle"].reshape(-1)
        current_node = td["current_node"].reshape(-1) #Get one current node index per attempt
        batch_indices = torch.arange(vehicle_features.shape[0], device=embeddings.device)

        vehicle_node_indices = torch.zeros_like(remaining, dtype=torch.long)
        vehicle_node_indices[batch_indices, current_vehicle] = current_node
        
        vehicle_embeddings = self.fleet_encoder(
            vehicle_features=vehicle_features,
            node_embeddings=node_embeddings,
            vehicle_node_indices=vehicle_node_indices,
            remaining=remaining,
            visited=visited,
        )
        fleet_context = vehicle_embeddings[batch_indices, current_vehicle] #Select the active vehicle’s embedding for each attempt.
        #batch_shape has instance dim and start dim, 
        # so our output has instances, starts, embeddings
        return fleet_context.reshape(*batch_shape, self.embed_dim)