from transformers import GPT2Model
import plotly.express as px
from sklearn.decomposition import PCA


def main():
    model = GPT2Model.from_pretrained("gpt2")

    # 3.a — matrice des embeddings positionnels (learned)
    # GPT-2 : model.wpe est la table d'embeddings de position
    position_embeddings = model.wpe.weight  # shape: (n_positions, n_embd)

    print("Shape position embeddings:", tuple(position_embeddings.size()))
    print("n_embd:", model.config.n_embd)
    print("n_positions:", model.config.n_positions)

    # 3.b — PCA sur positions 0..49
    positions_50 = position_embeddings[:50].detach().cpu().numpy()
    pca_50 = PCA(n_components=2)
    reduced_50 = pca_50.fit_transform(positions_50)

    fig50 = px.scatter(
        x=reduced_50[:, 0],
        y=reduced_50[:, 1],
        text=[str(i) for i in range(len(reduced_50))],
        color=list(range(len(reduced_50))),
        title="Encodages positionnels GPT-2 (PCA, positions 0-50)",
        labels={"x": "PCA 1", "y": "PCA 2"}
    )
    fig50.write_html("TP1/positions_50.html")
    print("Saved: TP1/positions_50.html")

    # 3.c — PCA sur positions 0..200
    positions_200 = position_embeddings[:201].detach().cpu().numpy()  # 0..200 inclus => 201 points
    pca_200 = PCA(n_components=2)
    reduced_200 = pca_200.fit_transform(positions_200)

    fig200 = px.scatter(
        x=reduced_200[:, 0],
        y=reduced_200[:, 1],
        text=[str(i) for i in range(len(reduced_200))],
        color=list(range(len(reduced_200))),
        title="Encodages positionnels GPT-2 (PCA, positions 0-200)",
        labels={"x": "PCA 1", "y": "PCA 2"}
    )
    fig200.write_html("TP1/positions_200.html")
    print("Saved: TP1/positions_200.html")


if __name__ == "__main__":
    main()
