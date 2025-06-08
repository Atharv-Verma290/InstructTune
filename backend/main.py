# --- Imports ---
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel, Field
from time import time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# --- Load Environment Variables ---
load_dotenv()

# --- FastAPI App Setup ---
app = FastAPI()

# --- Model Initialization ---
openai_model = ChatOpenAI(model="gpt-4o-mini")
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- State Definitions ---
class OverallState(TypedDict):
    input_example: str
    output_example: str 
    original_prompt: str
    context: str 
    candidate_prompts: List[str]
    outputs: Dict[str, Dict[str, Any]]
    evaluations: Dict[str, Dict[str, Any]]
    optimized_prompt: str
    prompt_to_test: str
    dashboard_data: str 
    total_tokens: str 
    feedback: str 
    prompt_versions: List[str]
    current_version: str
    counter: int
    models_list: List[Dict[str, str]]


class TesterState(TypedDict):
    input_prompt: str
    instruction: str 
    models_list: List[Dict[str, str]]
    model_0_output: Dict[str, Any] 
    model_1_output: Dict[str, Any] 
    model_2_output: Dict[str, Any]


class EvaluatorState(TypedDict):
    instruction: str 
    input_prompt: str
    outputs: Dict[str, Dict[str, Any]]
    context: str 
    models_list: List[Dict[str, str]]
    model_0_eval: Dict[str, Any]
    model_1_eval: Dict[str, Any]
    model_2_eval: Dict[str, Any]


# --- Pydantic Models ---
class EvaluationScores(BaseModel):
    relevance: float = Field(..., ge=0, le=1, description="The relevance score of the output based on the prompt and context")
    clarity_readability: float = Field(..., ge=0, le=1, description="The clarity or readability score of the output based on the prompt and context")
    tone_match: float = Field(..., ge=0, le=1, description="Score how well does the tone of the output match based on the prompt and context")
    efficiency: float = Field(..., ge=0, le=1, description="The efficiency score of the output based on the prompt and context")
    precision: float = Field(..., ge=0, le=1, description="The precision score of the output based on the prompt and context")
    overall_score: float = Field(..., ge=0, le=1, description="The overall score of the output based on the prompt and context")
    feedback: str = Field(..., description="Feedback on the prompt and suggest changes to the prompt")


# --- Prompts ---
EVALUATOR_PROMPT="""
You are an evaluation assistant.

Given the context of the task(if given) and the model's output, score the output on:

{evaluation_schema}

### Context:
{context}

### Example Input:
{input_prompt}

### Prompt:
{instruction}

### Model Output:
{output}
"""

OPTIMIZER_PROMPT="""
You are a expert prompt engineer. Your job is optimize the given prompt based on the context and evaluation.ONLY OUTPUT THE OPTIMIZED PROMPT. DO NOT INCLUDE THE TOPIC FROM THE EVALUATION INTO THE NEW PROMPT.

### Prompt: 
{prompt}

### Context:
{context}

### Evaluation:
{evaluations}
"""


# --- Nodes ---
def prompt_selector_node(state: OverallState):
    prompt_to_test = state.get("optimized_prompt") or state.get("original_prompt")
    return {"prompt_to_test": prompt_to_test}


def model_0_tester_node(state: TesterState):
    input_prompt = state.get('input_prompt', '')
    instruction = state.get('instruction', '')
    models_list = state.get('models_list', {})
    if len(models_list) <= 0:
        return {}  # No model, skip
    model_info = models_list[0]
    model_name = next(iter(model_info))
    provider = model_info[model_name]

    if model_name and provider:
        model = init_chat_model(model=model_name, model_provider=provider)
        messages = [
            SystemMessage(content=instruction),
            HumanMessage(content=input_prompt)
        ]
        start_time = time()
        response = model.invoke(messages)
        end_time = time()

        model_0_output = {}
        model_0_output["output"] = response.content 
        model_0_output["tokens"] = response.usage_metadata["total_tokens"]
        model_0_output["latency"] = round((end_time - start_time), 2)
    
    return {"model_0_output": model_0_output}


def model_1_tester_node(state: TesterState):
    input_prompt = state.get('input_prompt', '')
    instruction = state.get('instruction', '')
    models_list = state.get('models_list', {})
    if len(models_list) <= 1:
        return {}  # No second model, skip
    model_info = models_list[1]
    model_name = next(iter(model_info))
    provider = model_info[model_name]

    if model_name and provider:
        model = init_chat_model(model=model_name, model_provider=provider)
        messages = [
            SystemMessage(content=instruction),
            HumanMessage(content=input_prompt)
        ]
        start_time = time()
        response = model.invoke(messages)
        end_time = time()

        model_1_output = {}
        model_1_output["output"] = response.content 
        model_1_output["tokens"] = response.usage_metadata["total_tokens"]
        model_1_output["latency"] = round((end_time - start_time), 2)
    
    return {"model_1_output": model_1_output}


def model_2_tester_node(state: TesterState):
    input_prompt = state.get('input_prompt', '')
    instruction = state.get('instruction', '')
    models_list = state.get('models_list', {})
    if len(models_list) <= 2:
        return {}  # No third model, skip
    model_info = models_list[2]
    model_name = next(iter(model_info))
    provider = model_info[model_name]

    if model_name and provider:
        model = init_chat_model(model=model_name, model_provider=provider)
        messages = [
            SystemMessage(content=instruction),
            HumanMessage(content=input_prompt)
        ]
        start_time = time()
        response = model.invoke(messages)
        end_time = time()

        model_2_output = {}
        model_2_output["output"] = response.content 
        model_2_output["tokens"] = response.usage_metadata["total_tokens"]
        model_2_output["latency"] = round((end_time - start_time), 2)
    
    return {"model_2_output": model_2_output}


def prompt_tester_node(state: OverallState):
    prompt = state.get('prompt_to_test')
    input = state.get('input_example', '')
    version = state.get('current_version', 'original')
    outputs = state.get('outputs', {})
    models_list = state.get('models_list', {})

    graph_output = prompt_tester_graph.invoke({"input_prompt": input, "instruction": prompt, "models_list": models_list})

    for idx, model_info in enumerate(models_list):
        model_name = next(iter(model_info))
        result = graph_output.get(f"model_{idx}_output", {})
        if model_name not in outputs:
            outputs[model_name] = {}
        outputs[model_name][version] = result

    return {"outputs": outputs}


def model_0_evaluator_node(state: EvaluatorState):
    instruction = state.get('instruction', '')
    input_prompt = state.get('input_prompt', '')
    outputs = state.get('outputs', {})
    models_list = state.get('models_list', [])
    if len(models_list) == 0:
        return {}
    model_info = models_list[0]
    model_name = next(iter(model_info))
    provider = model_info[model_name]
    model_outputs = outputs.get(model_name, {})
    if not model_outputs:
        return {}
    version = list(model_outputs.keys())[0]
    context = state.get('context', '')
    output_data = model_outputs[version]

    messages = [
        HumanMessage(
            content=EVALUATOR_PROMPT.format(
                evaluation_schema=EvaluationScores.model_json_schema(),
                context=context, 
                input_prompt=input_prompt, 
                instruction=instruction,
                output=output_data.get('output', '')
            )
        ),
    ]
    model = init_chat_model(model=model_name, model_provider=provider)
    structured_model = model.with_structured_output(EvaluationScores, method="function_calling")
    response = structured_model.invoke(messages)
    # evaluation = response.model_dump_json()
    return {"model_0_eval": response.model_dump()}


def model_1_evaluator_node(state: EvaluatorState):
    instruction = state.get('instruction', '')
    input_prompt = state.get('input_prompt', '')
    outputs = state.get('outputs', {})
    models_list = state.get('models_list', [])
    if len(models_list) <= 1:
        return {}
    model_info = models_list[1]
    model_name = next(iter(model_info))
    provider = model_info[model_name]
    model_outputs = outputs.get(model_name, {})
    if not model_outputs:
        return {}
    version = list(model_outputs.keys())[0]
    context = state.get('context', '')
    output_data = model_outputs[version]

    messages = [
        HumanMessage(
            content=EVALUATOR_PROMPT.format(
                evaluation_schema=EvaluationScores.model_json_schema(), 
                context=context, 
                input_prompt=input_prompt, 
                instruction=instruction,
                output=output_data.get('output', '')
            )
        ),
    ]

    model = init_chat_model(model=model_name, model_provider=provider)
    structured_model = model.with_structured_output(EvaluationScores, method="function_calling")
    response = structured_model.invoke(messages)
    # evaluation = response.model_dump
    return {"model_1_eval": response.model_dump()}


def model_2_evaluator_node(state: EvaluatorState):
    instruction = state.get('instruction', '')
    input_prompt = state.get('input_prompt', '')
    outputs = state.get('outputs', {})
    models_list = state.get('models_list', [])
    if len(models_list) <= 2:
        return {}
    model_info = models_list[2]
    model_name = next(iter(model_info))
    provider = model_info[model_name]
    model_outputs = outputs.get(model_name, {})
    if not model_outputs:
        return {}
    version = list(model_outputs.keys())[0]
    context = state.get('context', '')
    output_data = model_outputs[version]

    messages = [
        HumanMessage(
            content=EVALUATOR_PROMPT.format(
                evaluation_schema=EvaluationScores.model_json_schema(), 
                context=context, 
                input_prompt=input_prompt, 
                instruction=instruction,
                output=output_data.get('output', '')
            )
        ),
    ]

    model = init_chat_model(model=model_name, model_provider=provider)
    structured_model = model.with_structured_output(EvaluationScores, method="function_calling")
    response = structured_model.invoke(messages)
    # evaluation = response.model_dump
    return {"model_2_eval": response.model_dump()}


def prompt_evaluator_node(state: OverallState):
    instruction = state.get('prompt_to_test', '')
    input_example = state.get('input_example', '')
    context = state.get('context')
    outputs = state.get('outputs', {})
    version = state.get('current_version', 'original')
    evaluations = state.get('evaluations', {})
    models_list = state.get('models_list', [])

    graph_input = {
        "instruction": instruction,
        "input_prompt": input_example,
        "context": context,
        "outputs": {
            model_name: {version: outputs.get(model_name, {}).get(version, {})}
            for model_info in models_list
            for model_name in model_info
        },
        "models_list": models_list
    }

    results = prompt_eval_graph.invoke(graph_input)

    for idx, model_info in enumerate(models_list):
        model_name = next(iter(model_info))
        eval_key = f"model_{idx}_eval"
        eval_data = results.get(eval_key)
        if eval_data is not None:
            if model_name not in evaluations:
                evaluations[model_name] = {}
            if hasattr(eval_data, "model_dump"):
                evaluations[model_name][version] = eval_data.model_dump()
            else:
                evaluations[model_name][version] = eval_data

    return {"evaluations": evaluations}


def optimizer_node(state: OverallState):
    previous_prompt = state.get('prompt_to_test', state.get('original_prompt'))
    context = state.get('context', '')
    evaluations = state.get('evaluations', {})
    current_version = state.get('current_version', 'original')
    counter = state.get("counter", 0)

    current_evals = {
        model: model_evals.get(current_version, {})
        for model, model_evals in evaluations.items()
        if current_version in model_evals
    }

    messages = [
        SystemMessage(content=OPTIMIZER_PROMPT.format(prompt=previous_prompt, context=context, evaluations=current_evals))
    ]
    response = openai_model.invoke(messages)
    new_prompt = response.content

    prompt_versions = state.get('prompt_versions', ['original'])
    new_version = f'optimized_{len(prompt_versions)}'

    updated_versions = prompt_versions + [new_version]
    updated_candidates = state.get("candidate_prompts", []) + [new_prompt]

    return {
        "optimized_prompt": new_prompt,
        "prompt_versions": updated_versions,
        "candidate_prompts": updated_candidates,
        "prompt_to_test": new_prompt,
        "current_version": new_version,
        "counter": counter + 1
    }


def routing_function(state: OverallState):
    SCORE_THRESHOLD = 0.9
    current_version = state.get('current_version', 'original')
    evaluations = state.get('evaluations', {})

    scores = []
    for model_evals in evaluations.values():
        eval_data = model_evals.get(current_version, {})
        score = eval_data.get("overall_score")
        if isinstance(score, (float, int)):
            scores.append(score)

    # If no scores are available, assume continue
    if not scores:
        return True

    average_score = sum(scores)/len(scores)
    max_retries = 2
    return average_score < SCORE_THRESHOLD


# --- Graph Definitions ---
prompt_tester_workflow = StateGraph(TesterState)
prompt_tester_workflow.add_node("model_0_tester", model_0_tester_node)
prompt_tester_workflow.add_node("model_1_tester", model_1_tester_node)
prompt_tester_workflow.add_node("model_2_tester", model_2_tester_node)
prompt_tester_workflow.set_entry_point("model_0_tester")
prompt_tester_workflow.set_entry_point("model_1_tester")
prompt_tester_workflow.set_entry_point("model_2_tester")
prompt_tester_workflow.add_edge("model_0_tester", END)
prompt_tester_workflow.add_edge("model_1_tester", END)
prompt_tester_workflow.add_edge("model_2_tester", END)
prompt_tester_graph = prompt_tester_workflow.compile()

prompt_eval_workflow  = StateGraph(EvaluatorState)
prompt_eval_workflow.add_node("model_0_evaluator", model_0_evaluator_node)
prompt_eval_workflow.add_node("model_1_evaluator", model_1_evaluator_node)
prompt_eval_workflow.add_node("model_2_evaluator", model_2_evaluator_node)
prompt_eval_workflow.set_entry_point("model_0_evaluator")
prompt_eval_workflow.set_entry_point("model_1_evaluator")
prompt_eval_workflow.set_entry_point("model_2_evaluator")
prompt_eval_workflow.add_edge("model_0_evaluator", END)
prompt_eval_workflow.add_edge("model_1_evaluator", END)
prompt_eval_workflow.add_edge("model_2_evaluator", END)
prompt_eval_graph = prompt_eval_workflow.compile()

graph_builder = StateGraph(OverallState)
graph_builder.add_node("prompt_selector", prompt_selector_node)
graph_builder.add_node("prompt_tester", prompt_tester_node)
graph_builder.add_node("prompt_evaluator", prompt_evaluator_node)
graph_builder.add_node("prompt_optimizer", optimizer_node)
graph_builder.set_entry_point("prompt_selector")
graph_builder.add_edge("prompt_selector", "prompt_tester")
graph_builder.add_edge("prompt_tester", "prompt_evaluator")
graph_builder.add_conditional_edges("prompt_evaluator", routing_function, {True: "prompt_optimizer", False: END})
graph_builder.add_edge("prompt_optimizer", "prompt_tester")
graph = graph_builder.compile()


class GraphInput(BaseModel):
    original_prompt: str
    input_example: str 
    context: str 
    models_list: list 

@app.post("/run-graph")
async def run_graph(input_data: GraphInput):
    result = graph.invoke(input_data)
    return JSONResponse(content=result)

# --- Main Execution ---
# if __name__ == "__main__":
#     models_list = [
#         {"gpt-4o-mini": "openai"},
#         {"gemini-2.0-flash": "google-genai"}
#     ]
#     inputs = {
#         "original_prompt": "Teach the given topic from the user.",
#         "input_example": "What is gravity?",
#         "context": "A middle school teaching agent used to teach complex topics to middle school kids.",
#         "models_list": models_list
#     }

#     result = graph.invoke(inputs)