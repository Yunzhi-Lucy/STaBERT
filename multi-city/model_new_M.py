import torch
import torch.nn as nn


def create_poi_mask(poi_categories, pad_value=0):

    return (poi_categories != pad_value).int()

class DayEmbeddingModel(nn.Module):
    def __init__(self, embed_size):
        super(DayEmbeddingModel, self).__init__()
        self.day_embedding = nn.Embedding(num_embeddings=76, embedding_dim=embed_size)

    def forward(self, day):
        return self.day_embedding(day)


class TimeEmbeddingModel(nn.Module):
    def __init__(self, embed_size):
        super(TimeEmbeddingModel, self).__init__()
        self.time_embedding = nn.Embedding(num_embeddings=49, embedding_dim=embed_size)

    def forward(self, time):
        return self.time_embedding(time)


class LocationXEmbeddingModel(nn.Module):
    def __init__(self, embed_size):
        super(LocationXEmbeddingModel, self).__init__()
        self.location_embedding = nn.Embedding(num_embeddings=202, embedding_dim=embed_size)

    def forward(self, location):
        return self.location_embedding(location)


class LocationYEmbeddingModel(nn.Module):
    def __init__(self, embed_size):
        super(LocationYEmbeddingModel, self).__init__()
        self.location_embedding = nn.Embedding(num_embeddings=202, embedding_dim=embed_size)

    def forward(self, location):
        return self.location_embedding(location)


class TimedeltaEmbeddingModel(nn.Module):
    def __init__(self, embed_size):
        super(TimedeltaEmbeddingModel, self).__init__()
        self.timedelta_embedding = nn.Embedding(num_embeddings=48, embedding_dim=embed_size)

    def forward(self, timedelta):
        return self.timedelta_embedding(timedelta)


class CityEmbeddingModel(nn.Module):
    def __init__(self, embed_size):
        super(CityEmbeddingModel, self).__init__()
        self.city_embedding = nn.Embedding(num_embeddings=5, embedding_dim=embed_size)

    def forward(self, city):
        return self.city_embedding(city)
    

class POIEmbeddingModel(nn.Module):
    def __init__(self, embed_size):
        super(POIEmbeddingModel, self).__init__()
        self.poi_embedding = nn.Embedding(num_embeddings=86, embedding_dim=embed_size)

    def forward(self, poi_categories):
        return self.poi_embedding(poi_categories)

class POIcountEmbeddingModel(nn.Module):
    def __init__(self, embed_size):
        super(POIcountEmbeddingModel, self).__init__()
        self.poi_embedding = nn.Embedding(num_embeddings=736, embedding_dim=embed_size)

    def forward(self, poi_counts):
        return self.poi_embedding(poi_counts)

class POIMeanPoolingModel(nn.Module):
    def __init__(self):
        super(POIMeanPoolingModel, self).__init__()

    def forward(self, poi_embeddings, poi_mask):

        valid_poi_count = poi_mask.sum(dim=2, keepdim=True).clamp(min=1)  # [batch, seq_len, 1]


        poi_embeddings = poi_embeddings * poi_mask.unsqueeze(-1)  # [batch, seq_len, num_pois, embed_size]
        mean_poi_embedding = poi_embeddings.sum(dim=2) / valid_poi_count  # [batch, seq_len, embed_size]

        return mean_poi_embedding


class EmbeddingLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(EmbeddingLayer, self).__init__()

        self.day_embedding = DayEmbeddingModel(embed_size)
        self.time_embedding = TimeEmbeddingModel(embed_size)
        self.location_x_embedding = LocationXEmbeddingModel(embed_size)
        self.location_y_embedding = LocationYEmbeddingModel(embed_size)
        self.timedelta_embedding = TimedeltaEmbeddingModel(embed_size)
        self.city_embedding = CityEmbeddingModel(embed_size)

        self.poi_embedding = POIEmbeddingModel(embed_size)
        self.poi_count_embedding = POIcountEmbeddingModel(embed_size)

        self.poi_pooling = POIMeanPoolingModel()

    def forward(self, day, time, location_x, location_y, timedelta, poi_categories, poi_counts, city):

        day_embed = self.day_embedding(day)
        time_embed = self.time_embedding(time)
        location_x_embed = self.location_x_embedding(location_x)
        location_y_embed = self.location_y_embedding(location_y)
        timedelta_embed = self.timedelta_embedding(timedelta)
        city_embed = self.city_embedding(city)


        poi_embed = self.poi_embedding(poi_categories)  # [batch, seq_len, num_pois, embed_dim]
        poi_count_embed = self.poi_count_embedding(poi_counts)  # [batch, seq_len, num_pois, embed_dim]


        poi_mask = create_poi_mask(poi_categories)  # [batch, seq_len, num_pois]


        poi_full_embed = poi_embed + poi_count_embed
        poi_final = self.poi_pooling(poi_full_embed, poi_mask)  # [batch, seq_len, embed_dim]



        embed = day_embed + time_embed + location_x_embed + location_y_embed + timedelta_embed + poi_final + city_embed
        return embed


class TransformerEncoderModel(nn.Module):
    def __init__(self, layers_num, heads_num, embed_size):
        super(TransformerEncoderModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads_num)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=layers_num)

    def forward(self, input, src_key_padding_mask):
        return self.transformer_encoder(input, src_key_padding_mask=src_key_padding_mask)





class FFNLayer(nn.Module):
    def __init__(self, embed_size):
        super(FFNLayer, self).__init__()
        self.ffn1 = nn.Sequential(nn.Linear(embed_size, 16), nn.ReLU(), nn.Linear(16, 200))
        self.ffn2 = nn.Sequential(nn.Linear(embed_size, 16), nn.ReLU(), nn.Linear(16, 200))

    def forward(self, input):
        output_x = self.ffn1(input)
        output_y = self.ffn2(input)
        return torch.stack([output_x, output_y], dim=-2)


class LPBERT(nn.Module):
    def __init__(self, layers_num, heads_num, embed_size):
        super(LPBERT, self).__init__()
        self.embedding_layer = EmbeddingLayer(embed_size, heads_num)
        self.transformer_encoder = TransformerEncoderModel(layers_num, heads_num, embed_size)
        self.ffn_layer = FFNLayer(embed_size)

    def forward(self, day, time, location_x, location_y, timedelta, len, poi_categories, poi_counts, city):
        embed = self.embedding_layer(day, time, location_x, location_y, timedelta, poi_categories, poi_counts, city)
        embed = embed.transpose(0, 1)

        max_len = day.shape[-1]
        indices = torch.arange(max_len, device=len.device).unsqueeze(0)
        src_key_padding_mask = ~(indices < len.unsqueeze(-1))

        transformer_encoder_output = self.transformer_encoder(embed, src_key_padding_mask)
        transformer_encoder_output = transformer_encoder_output.transpose(0, 1)

        output = self.ffn_layer(transformer_encoder_output)
        return output
