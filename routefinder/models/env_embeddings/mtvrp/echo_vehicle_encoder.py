import torch.nn as nn
import torch
#from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.attention import MultiHeadAttention, MultiHeadCrossAttention

class EchoVehicleEncoder(nn.Module):
    """ECHO-inspired vehicle encoder, starting with its vehicle MLP."""
    def __init__(self, embed_dim=128, num_heads=8):
        super().__init__()

        self.veh_encoder_mlp = nn.Sequential(
            nn.Linear(4, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Location fusion and attention layers next
        self.veh_encoder_w = nn.Linear(2 * embed_dim, embed_dim)

        self.veh_encoder_self_attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=False,
        )
        self.veh_encoder_cross_attention = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=True,
        )




    def forward(self, vehicle_features, node_embeddings, vehicle_node_indices, remaining, visited):
        vehicle_embeddings = self.veh_encoder_mlp(vehicle_features)

        batch_indices = torch.arange(
            vehicle_features.shape[0],
            device=node_embeddings.device,
        )

        # Retrieve the node embedding at each vehicle's current location.
        #section same as echo
        #node_embeddings would contain the embeddings for every node in every instance: [batch, nodes, 128]
        #batch_indices identifies the instances
        #[:, None] we keep all instances with ":" and add a dimension
        #vehicle_node_indices tells which node is ocuppied for each vehicle
        #so we selected the correct instance and node for each vehicle
        vehicle_node_embeddings = node_embeddings[
            batch_indices[:, None], vehicle_node_indices  ] 
        #contains the embedding of the node where each vehicle is currently located

        vehicle_embeddings = self.veh_encoder_w(
            torch.cat([vehicle_node_embeddings, vehicle_embeddings], dim=-1,  ))

        #collect infor from available vehicles
        attention_output = self.veh_encoder_self_attention( vehicle_embeddings,  attn_mask=remaining, )

        #add fleet information to the vehicle representations
        vehicle_embeddings = vehicle_embeddings + attention_output


        # Attend to unvisited customers and always keep the depot available, as in ECHO.
        available_nodes = ~visited
        available_nodes[:, 0] = True

        #create a summary o node info for each vehicle
        node_context = self.veh_encoder_cross_attention(
            vehicle_embeddings,
            node_embeddings,
            cross_attn_mask=available_nodes.unsqueeze(1),
        )
        vehicle_embeddings = vehicle_embeddings + node_context


        #set unavailable vehicles output embeddings to zero.
        vehicle_embeddings = vehicle_embeddings.masked_fill(
            ~remaining.unsqueeze(-1), 0
        )

        return vehicle_embeddings