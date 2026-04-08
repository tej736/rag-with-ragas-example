import asyncio
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

# Add the root directory to the sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from dotenv import load_dotenv

from app.config import normalize_provider, validate_distributions
from app.eval import Eval
from app.rag import Rag
from app.tracking import DatasetRegistry, ExperimentTracker

load_dotenv()

OPENAI_EMBED_MODELS = ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]
OPENAI_LLM_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
HF_EMBED_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
HF_LLM_MODELS = ["HuggingFaceH4/zephyr-7b-beta", "mistralai/Mistral-7B-Instruct-v0.2"]


class RagWithRagasApp:
    def __init__(self):
        st.set_page_config(
            page_title="LLM Evaluation & RAG Experimentation Lab",
            page_icon=":robot_face:",
            layout="wide",
        )
        self.rag = Rag()
        self.eval = Eval()
        self.tracker = ExperimentTracker()
        self.dataset_registry = DatasetRegistry()

    def _init_state(self):
        defaults = {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "hf_api_token": os.getenv("HUGGINGFACE_API_TOKEN", ""),
            "urls": [],
            "uploaded_files": [],
            "query": "",
            "run_results": [],
            "token_encoding_model": "gpt-4",
            "chunk_size": 800,
            "overlap": 400,
            "top_k": 3,
            "test_size": 4,
            "distributions": {"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25},
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _models_for_provider(self, provider: str):
        provider = normalize_provider(provider)
        if provider == "openai":
            return OPENAI_LLM_MODELS, OPENAI_EMBED_MODELS
        return HF_LLM_MODELS, HF_EMBED_MODELS

    def sidebar(self):
        st.sidebar.title("Lab Settings ⚙️")

        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            value=st.session_state.get("openai_api_key", ""),
        )
        hf_api_token = st.sidebar.text_input(
            "Hugging Face API Token",
            type="password",
            placeholder="hf_...",
            value=st.session_state.get("hf_api_token", ""),
        )

        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.session_state.openai_api_key = openai_api_key
        if hf_api_token:
            os.environ["HUGGINGFACE_API_TOKEN"] = hf_api_token
            st.session_state.hf_api_token = hf_api_token

        st.sidebar.title("Load Data 📤")

        uploaded_files = st.sidebar.file_uploader(
            "Upload data files", type=["doc", "pdf", "txt"], accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("app/data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
            st.sidebar.success("Files uploaded successfully")
            st.session_state.uploaded_files = uploaded_files

        url_input = st.sidebar.text_input("Enter a URL")
        if st.sidebar.button("Add URL") and url_input:
            st.session_state.urls.append(url_input)

        if st.session_state.urls:
            st.sidebar.caption("Registered URLs")
            st.sidebar.write("\n".join(st.session_state.urls))

        if st.sidebar.button("Clear output artifacts"):
            self.rag.clear_output_folder()
            st.sidebar.success("Cleared app/output JSON/CSV files")

    async def _build_index(self, provider: str, embedding_model: str, token_model: str, chunk_size: int, overlap: int):
        all_chunks = []
        for uploaded_file in st.session_state.uploaded_files:
            file_path = os.path.join("app/data", uploaded_file.name)
            if uploaded_file.name.endswith(".txt"):
                text = self.rag.load_text_file(file_path)
            elif uploaded_file.name.endswith(".pdf"):
                text = self.rag.load_pdf_file(file_path)
            else:
                continue
            all_chunks.extend(
                self.rag.process_text(
                    text=text,
                    token_encoding_model=token_model,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

        for url in st.session_state.urls:
            text = await self.rag.fetch_text_from_url(url)
            all_chunks.extend(
                self.rag.process_text(
                    text=text,
                    token_encoding_model=token_model,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

        self.rag.save_chunks_to_file(all_chunks, filename="app/output/data_chunks.json")

        embedded_chunks = await self.rag.embed_text_chunks(
            chunks=all_chunks,
            embedding_model=embedding_model,
            embedding_provider=provider,
        )
        self.rag.save_chunks_to_file(embedded_chunks, filename="app/output/embeddings.json")
        return embedded_chunks

    async def _run_query(
        self,
        label: str,
        query: str,
        llm_provider: str,
        llm_model: str,
        embedding_provider: str,
        embedding_model: str,
        token_model: str,
        chunk_size: int,
        overlap: int,
        top_k: int,
    ):
        embedded_chunks = await self._build_index(
            provider=embedding_provider,
            embedding_model=embedding_model,
            token_model=token_model,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        query_embedding = await self.rag.embed_query(
            query=query,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
        )
        top_chunks_with_scores = self.rag.cosine_similarity_search(
            query_embedding=query_embedding,
            embedded_chunks=embedded_chunks,
            top_k=top_k,
            return_scores=True,
        )
        top_chunks = [x["text"] for x in top_chunks_with_scores]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Here are some documents that may help answer the user query: {top_chunks}. Please provide an answer to the query only based on the documents. If the documents don't contain the answer, say that you don't know.\n\nquery: {query}",
            },
        ]

        answer_result = await self.rag.generate_answer(
            messages=messages,
            model=llm_model,
            provider=llm_provider,
        )

        avg_similarity = (
            sum(item["score"] for item in top_chunks_with_scores) / len(top_chunks_with_scores)
            if top_chunks_with_scores
            else 0.0
        )

        run_payload = {
            "run_type": "rag_query",
            "run_mode": label,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "token_encoding_model": token_model,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "query": query,
            "retrieved_chunks_count": len(top_chunks),
            "avg_similarity": round(avg_similarity, 4),
            "latency_seconds": round(answer_result["latency_seconds"], 4),
            "prompt_tokens": answer_result.get("usage", {}).get("prompt_tokens", ""),
            "completion_tokens": answer_result.get("usage", {}).get("completion_tokens", ""),
            "total_tokens": answer_result.get("usage", {}).get("total_tokens", ""),
            "answer": answer_result["text"],
            "answer_preview": (answer_result["text"] or "")[:400],
        }
        tracked = self.tracker.log_run(run_payload)

        return {
            "tracked": tracked,
            "answer": answer_result["text"],
            "chunks": top_chunks_with_scores,
        }

    def _register_dataset_version(self, chunk_size: int, overlap: int):
        file_names = [f.name for f in st.session_state.uploaded_files]
        payload = {
            "source_file_count": len(file_names),
            "source_files": ",".join(file_names),
            "source_url_count": len(st.session_state.urls),
            "source_urls": ",".join(st.session_state.urls),
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
        return self.dataset_registry.register(payload)

    def _render_run_history(self):
        st.subheader("Run History & Analytics")
        runs_path = "app/output/experiments/runs.csv"
        if not os.path.exists(runs_path):
            st.info("No run history yet.")
            return

        runs_df = pd.read_csv(runs_path)
        st.dataframe(runs_df.tail(100), use_container_width=True)

        query_runs = runs_df[runs_df["run_type"] == "rag_query"].copy()
        if not query_runs.empty and "latency_seconds" in query_runs.columns:
            fig = px.box(
                query_runs,
                x="llm_provider",
                y="latency_seconds",
                color="llm_model",
                title="Latency by Provider/Model",
            )
            st.plotly_chart(fig, use_container_width=True)

        eval_runs = runs_df[runs_df["run_type"] == "ragas_eval"].copy()
        if not eval_runs.empty and "faithfulness_mean" in eval_runs.columns:
            eval_runs["timestamp"] = pd.to_datetime(eval_runs["timestamp"])
            fig = px.line(
                eval_runs.sort_values("timestamp"),
                x="timestamp",
                y=[
                    "context_precision_mean",
                    "context_recall_mean",
                    "faithfulness_mean",
                    "answer_relevancy_mean",
                ],
                title="RAGAS Metric Drift Across Runs",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Failure-case Explorer")
        answer_series = (
            query_runs["answer"].astype(str)
            if "answer" in query_runs.columns
            else pd.Series(dtype=str)
        )
        failures = query_runs[
            answer_series.str.lower().str.contains(
                "don't know|do not know", regex=True, na=False
            )
        ]
        if failures.empty:
            st.caption("No explicit 'don't know' responses logged yet.")
        else:
            st.dataframe(failures.tail(50), use_container_width=True)

        st.subheader("Export Reports")
        st.download_button(
            "Download Experiment Runs CSV",
            data=open(runs_path, "rb").read(),
            file_name="runs.csv",
            mime="text/csv",
        )

        summary = f"""# LLM Evaluation & RAG Experimentation Report

Generated at: {datetime.now(timezone.utc).isoformat()}

## Overview
- Total runs: {len(runs_df)}
- Query runs: {len(query_runs)}
- Eval runs: {len(eval_runs)}

## Current Focus
- Side-by-side provider comparison (OpenAI vs Hugging Face)
- RAGAS quality tracking over time
- SQL-friendly experiment artifacts
"""
        st.download_button(
            "Download Markdown Summary",
            data=summary,
            file_name="experiment-summary.md",
            mime="text/markdown",
        )

    def main_section(self):
        st.title("LLM Evaluation & RAG Experimentation Lab")
        st.caption("Portfolio branch: provider-agnostic RAG + RAGAS analytics + experiment tracking")

        tabs = st.tabs(["RAG Lab", "RAGAS Eval", "Analytics"])

        with tabs[0]:
            st.subheader("RAG Query Lab")
            compare_mode = st.checkbox("Enable side-by-side comparison", value=True)

            token_model = st.selectbox("Token Encoding Model", ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], index=1)
            chunk_size = st.slider("Chunk Size", 100, 2000, st.session_state.chunk_size)
            overlap = st.slider("Overlap", 0, 1000, st.session_state.overlap)
            top_k = st.slider("Top-K", 1, 10, st.session_state.top_k)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Config A")
                llm_provider_a = st.selectbox("LLM Provider A", ["openai", "huggingface"], key="llm_provider_a")
                llm_models_a, emb_models_a = self._models_for_provider(llm_provider_a)
                llm_model_a = st.selectbox("LLM Model A", llm_models_a, key="llm_model_a")
                emb_provider_a = st.selectbox("Embedding Provider A", ["openai", "huggingface"], key="emb_provider_a")
                _, emb_models_provider_a = self._models_for_provider(emb_provider_a)
                emb_model_a = st.selectbox("Embedding Model A", emb_models_provider_a, key="emb_model_a")

            if compare_mode:
                with c2:
                    st.markdown("##### Config B")
                    llm_provider_b = st.selectbox("LLM Provider B", ["openai", "huggingface"], key="llm_provider_b")
                    llm_models_b, emb_models_b = self._models_for_provider(llm_provider_b)
                    llm_model_b = st.selectbox("LLM Model B", llm_models_b, key="llm_model_b")
                    emb_provider_b = st.selectbox("Embedding Provider B", ["openai", "huggingface"], key="emb_provider_b")
                    _, emb_models_provider_b = self._models_for_provider(emb_provider_b)
                    emb_model_b = st.selectbox("Embedding Model B", emb_models_provider_b, key="emb_model_b")
            else:
                llm_provider_b = llm_model_b = emb_provider_b = emb_model_b = None

            query = st.text_area("Query", value=st.session_state.query)

            if st.button("Run Query"):
                if not st.session_state.uploaded_files and not st.session_state.urls:
                    st.warning("Upload at least one file or add one URL before running.")
                else:
                    st.session_state.query = query
                    dataset_record = self._register_dataset_version(
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    st.caption(f"Dataset version registered: {dataset_record['dataset_version']}")

                    with st.spinner("Running experiment(s)..."):
                        result_a = asyncio.run(
                            self._run_query(
                                label="A",
                                query=query,
                                llm_provider=llm_provider_a,
                                llm_model=llm_model_a,
                                embedding_provider=emb_provider_a,
                                embedding_model=emb_model_a,
                                token_model=token_model,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                top_k=top_k,
                            )
                        )

                        result_b = None
                        if compare_mode:
                            result_b = asyncio.run(
                                self._run_query(
                                    label="B",
                                    query=query,
                                    llm_provider=llm_provider_b,
                                    llm_model=llm_model_b,
                                    embedding_provider=emb_provider_b,
                                    embedding_model=emb_model_b,
                                    token_model=token_model,
                                    chunk_size=chunk_size,
                                    overlap=overlap,
                                    top_k=top_k,
                                )
                            )

                    if compare_mode and result_b:
                        left, right = st.columns(2)
                        with left:
                            st.markdown(f"### Config A ({llm_provider_a}:{llm_model_a})")
                            st.write(result_a["answer"])
                            with st.expander("Retrieved Contexts A"):
                                st.json(result_a["chunks"])
                        with right:
                            st.markdown(f"### Config B ({llm_provider_b}:{llm_model_b})")
                            st.write(result_b["answer"])
                            with st.expander("Retrieved Contexts B"):
                                st.json(result_b["chunks"])
                    else:
                        st.markdown(f"### Result ({llm_provider_a}:{llm_model_a})")
                        st.write(result_a["answer"])
                        with st.expander("Retrieved Contexts"):
                            st.json(result_a["chunks"])

        with tabs[1]:
            st.subheader("RAGAS Evaluation")

            llm_provider = st.selectbox("LLM Provider", ["openai", "huggingface"], key="eval_llm_provider")
            llm_models, emb_models = self._models_for_provider(llm_provider)
            llm_model = st.selectbox("LLM Model", llm_models, key="eval_llm_model")

            emb_provider = st.selectbox("Embedding Provider", ["openai", "huggingface"], key="eval_emb_provider")
            _, emb_models_specific = self._models_for_provider(emb_provider)
            emb_model = st.selectbox("Embedding Model", emb_models_specific, key="eval_emb_model")

            token_model = st.selectbox("Token Encoding Model", ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], key="eval_token_model", index=1)
            chunk_size = st.slider("Eval Chunk Size", 100, 2000, 800)
            overlap = st.slider("Eval Overlap", 0, 1000, 400)
            top_k = st.slider("Eval Top-K", 1, 10, 3)
            test_size = st.slider("Test Set Size", 1, 25, 4)

            d1, d2, d3 = st.columns(3)
            simple_dist = d1.slider("Simple", 0.0, 1.0, 0.5, 0.05)
            reasoning_dist = d2.slider("Reasoning", 0.0, 1.0, 0.25, 0.05)
            multi_context_dist = d3.slider("Multi-Context", 0.0, 1.0, 0.25, 0.05)
            distributions = {
                "simple": round(simple_dist, 2),
                "reasoning": round(reasoning_dist, 2),
                "multi_context": round(multi_context_dist, 2),
            }

            try:
                validate_distributions(distributions)
            except ValueError as ex:
                st.error(str(ex))

            if st.button("Start RAG Evaluation"):
                try:
                    validate_distributions(distributions)
                except ValueError as ex:
                    st.error(str(ex))
                else:
                    with st.spinner("Running evaluation..."):
                        asyncio.run(
                            self.eval.run_evaluation(
                                llm_model=llm_model,
                                token_encoding_model=token_model,
                                embedding_model=emb_model,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                top_k=top_k,
                                test_size=test_size,
                                distributions=distributions,
                                llm_provider=llm_provider,
                                embedding_provider=emb_provider,
                            )
                        )
                    st.success("Evaluation completed")

            if os.path.exists("app/output/testset.csv"):
                st.subheader("Test Set")
                st.dataframe(pd.read_csv("app/output/testset.csv"), use_container_width=True)

            if os.path.exists("app/output/evaluation_results.csv"):
                st.subheader("Evaluation Results")
                eval_df = pd.read_csv("app/output/evaluation_results.csv")
                st.dataframe(eval_df, use_container_width=True)

                metric_cols = [
                    "context_relevancy",
                    "context_precision",
                    "context_recall",
                    "faithfulness",
                    "answer_relevancy",
                ]
                available_cols = [c for c in metric_cols if c in eval_df.columns]
                if available_cols:
                    melted = eval_df[available_cols].reset_index().melt(id_vars="index", var_name="metric", value_name="score")
                    fig = px.bar(
                        melted,
                        x="metric",
                        y="score",
                        color="metric",
                        title="RAGAS Metrics by Question",
                        barmode="group",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            self._render_run_history()

    def render(self):
        self._init_state()
        self.sidebar()
        self.main_section()


if __name__ == "__main__":
    app = RagWithRagasApp()
    app.render()
