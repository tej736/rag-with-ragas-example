import asyncio
import os
import time

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    faithfulness,
)
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator
from termcolor import colored

from app.config import normalize_provider, validate_distributions
from app.rag import Rag
from app.tracking import ExperimentTracker
from dotenv import load_dotenv

load_dotenv()


class Eval:
    def __init__(self):
        self.rag = Rag()
        self.tracker = ExperimentTracker()

    async def run_evaluation(
        self,
        llm_model="gpt-4-turbo",
        token_encoding_model="gpt-4",
        embedding_model="text-embedding-3-large",
        chunk_size=800,
        overlap=400,
        top_k=3,
        test_size=4,
        distributions={"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25},
        llm_provider="openai",
        embedding_provider="openai",
    ):
        llm_provider = normalize_provider(llm_provider)
        embedding_provider = normalize_provider(embedding_provider)
        validate_distributions(distributions)

        data_folder_path = "app/data"
        print(colored("Starting to process files in the data folder...", "yellow"))

        all_chunks = await self.rag.process_files_in_folder_for_eval(
            data_folder_path=data_folder_path,
            token_encoding_model=token_encoding_model,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        print(colored("All chunks have been processed and saved.", "blue"))

        print(colored("Starting to generate a test set.", "green"))
        generator = TestsetGenerator.with_openai()
        testset = generator.generate_with_langchain_docs(
            documents=all_chunks,
            test_size=test_size,
            distributions={
                simple: distributions["simple"],
                reasoning: distributions["reasoning"],
                multi_context: distributions["multi_context"],
            },
        )

        testset_df = testset.to_pandas()
        os.makedirs("app/output", exist_ok=True)
        testset_df.to_csv("app/output/testset.csv", index=False)
        print(colored("Test set has been exported to app/output/testset.csv.", "blue"))

        questions = testset_df["question"].to_list()
        ground_truth = testset_df["ground_truth"].to_list()
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": ground_truth,
        }

        answer_latencies = []

        for query in questions:
            data["question"].append(query)
            query_embedding = await self.rag.embed_query(
                query=query,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
            )
            embedded_chunks = [
                {"text": chunk.page_content, "embedding": chunk.metadata["embedding"]}
                for chunk in all_chunks
            ]

            top_chunks = self.rag.cosine_similarity_search(
                query_embedding=query_embedding,
                embedded_chunks=embedded_chunks,
                top_k=top_k,
            )
            self.rag.save_top_chunks_text_to_file(
                top_chunks, filename="app/output/top_chunks.json"
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Here are some documents that may help answer the user query: {top_chunks}. Please provide an answer to the query only based on the documents. If the documents don't contain the answer, say that you don't know.\n\nquery: {query}",
                },
            ]

            result = await self.rag.generate_answer(
                messages=messages,
                model=llm_model,
                provider=llm_provider,
            )
            answer_latencies.append(result["latency_seconds"])
            data["answer"].append(result["text"])
            data["contexts"].append(top_chunks)

        dataset = Dataset.from_dict(data)
        dataset_df = dataset.to_pandas()
        dataset_df.to_csv("app/output/generated_dataset.csv", index=False)
        print(
            colored(
                "Generated dataset has been exported to app/output/generated_dataset.csv.",
                "blue",
            )
        )

        start_eval = time.perf_counter()
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_relevancy,
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        eval_latency = time.perf_counter() - start_eval

        result_df = result.to_pandas()
        result_df.to_csv("app/output/evaluation_results.csv", index=False)
        print(
            colored(
                "Evaluation results have been exported to app/output/evaluation_results.csv.",
                "blue",
            )
        )

        metric_summary = {
            "context_relevancy_mean": float(result_df["context_relevancy"].mean()),
            "context_precision_mean": float(result_df["context_precision"].mean()),
            "context_recall_mean": float(result_df["context_recall"].mean()),
            "faithfulness_mean": float(result_df["faithfulness"].mean()),
            "answer_relevancy_mean": float(result_df["answer_relevancy"].mean()),
        }
        avg_answer_latency = (
            sum(answer_latencies) / len(answer_latencies) if answer_latencies else 0.0
        )

        self.tracker.log_run(
            {
                "run_type": "ragas_eval",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k": top_k,
                "test_size": test_size,
                "eval_latency_seconds": round(eval_latency, 4),
                "avg_answer_latency_seconds": round(avg_answer_latency, 4),
                **metric_summary,
            }
        )

        return result_df


if __name__ == "__main__":
    eval = Eval()
    asyncio.run(
        eval.run_evaluation(
            llm_model="gpt-4-turbo",
            token_encoding_model="gpt-4",
            embedding_model="text-embedding-3-large",
            chunk_size=800,
            overlap=400,
            top_k=3,
            test_size=4,
            distributions={"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25},
            llm_provider="openai",
            embedding_provider="openai",
        )
    )
