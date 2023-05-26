import streamlit as st
import streamlit.components.v1 as components
from autofaiss.external.metadata import IndexMetadata
from autofaiss.external.optimize import get_optimal_index_keys_v2
from PIL import Image

# SETUP ------------------------------------------------------------------------
favicon = Image.open("favicon.ico")
st.set_page_config(
    page_title="Autofaiss knn indices selector",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="auto",
)


# Sidebar contents ------------------------------------------------------------------------
with st.sidebar:
    st.title("Autofaiss knn indices selector")
    st.markdown(
        """
    ## About
    This app automatically creates a Faiss knn indices with the most optimal similarity search parameters., built using:
    - [Streamlit](https://streamlit.io/)
    - [Autofaiss](https://criteo.github.io/autofaiss/)
    """
    )
    st.write(
        "Made with ❤️ by [Chasquilla Engineer](https://resume.chasquillaengineer.com/)"
    )


# ROW 1 ------------------------------------------------------------------------

Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 300vw 300vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
    </style> 
    
    <div class="title">
        <h1>Optimal Index Selector</h1>
    </div>
    """
components.html(Title_html)


def get_text(nb_vectors, dim_vectors, max_index_memory_usage):
    _nb_vectors = round(nb_vectors)
    _dim_vectors = round(dim_vectors)
    m = round(max_index_memory_usage)
    _max_index_memory_usage = f"{m}MB"  # if m < 1000 else f"{m/1000}GB"

    s = ""
    # s += f"nb_vectors={_nb_vectors}, dim_vectors={_dim_vectors}, max_index_memory_usage={_max_index_memory_usage}\n"
    opti_keys = get_optimal_index_keys_v2(
        _nb_vectors, _dim_vectors, _max_index_memory_usage
    )

    if not opti_keys:
        s += "Impossible to find index parameters for such a large dataset and low memory"
    else:
        index_metadada = IndexMetadata(opti_keys[0], _nb_vectors, _dim_vectors)

        s += f"The optimal index would be: {opti_keys[0]}"
        s += "\n\n"
        s += index_metadada.get_index_description(tunable_parameters_infos=True)

    return s


nb_vectors = st.slider("Number of vectors in the dataset", 0, 1000000, 10000)
dim_vectors = st.slider("Dimension of vectors in the dataset", 0, 4096, 2)
max_index_memory_usage = st.slider("Max size of index in MB", 0, 1000, 100)
st.write(str(get_text(nb_vectors, dim_vectors, max_index_memory_usage)))
