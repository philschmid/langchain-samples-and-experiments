import gradio as gr
import os
import arxiv
from langchain.document_loaders import PyPDFLoader
from anthropic import Anthropic
import tempfile

from langchain.chat_models import ChatAnthropic
from langchain.prompts import HumanMessagePromptTemplate

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[
        gr.themes.GoogleFont("Open Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)


# read env
os.environ = {
    **os.environ,
    **{env.split("=")[0]: env.split("=")[1].replace("\n", "") for env in open("../.env", "r").readlines()},
}

MAX_TOTAL_TOKENS = 30000
tokenizer = Anthropic()
llm = ChatAnthropic(temperature=0, max_tokens=1024)


prompt_template = HumanMessagePromptTemplate.from_template(
    """please complete the following tasks for the <paper>:
1. Extract the objective and contribution of the paper in one sentence. (In <objective> tags)
2. Extract the implementation details as step-by-step instructions focus on technical details. (In <implementation> tags)
3. Extract the key insights and learnings of the paper as bullet points. (In <insights> tags)
4. Extract the results of the paper in one sentence. (In <results> tags)

<paper>{paper}</paper>
"""
)


def get_paper_details(paper_id):
    # get paper by id
    with tempfile.TemporaryDirectory() as tmpdirname:
        paper = next(arxiv.Search(id_list=[paper_id]).results())
        # download paper
        path_to_file = os.path.join(tmpdirname, f"{paper_id}.pdf")
        downloaded_file = paper.download_pdf(filename=path_to_file)

        loader = PyPDFLoader(downloaded_file)
        pages = loader.load_and_split()

    # count tokens of the paper

    total_tokens = 0
    paper_content = ""

    for page in pages:
        tokens_per_page = tokenizer.count_tokens(page.page_content)
        total_tokens += tokens_per_page
        # add page content to paper content
        paper_content += page.page_content + "\n"

        # check if prompt got too long
        if total_tokens > MAX_TOTAL_TOKENS:
            break

    # format prompt
    prompt = prompt_template.format(paper=paper_content)

    # generate summary
    res = llm([prompt])

    # write markdown
    md = f"# {paper.title}\n\n"

    md += (
        res.content.replace("<summary>", "### Summary\n")
        .replace("</summary>", "")
        .replace("<objective>", "### Objective\n")
        .replace("</objective>", "")
        .replace("<implementation>", "### Implementation\n")
        .replace("</implementation>", "")
        .replace("<insights>", "### Insights\n")
        .replace("</insights>", "")
        .replace("<results>", "### Results\n")
        .replace("</results>", "")
    )
    return md


examples = ["2309.05463", "2306.01116"]

with gr.Blocks(theme=theme, analytics_enabled=False) as demo:
    paper_id = gr.Textbox(label="Arxiv paper ID", lines=1)
    output = gr.Markdown()
    create_btn = gr.Button("Get paper details")
    create_btn.click(fn=get_paper_details, inputs=paper_id, outputs=output)

    gr.Examples(
        examples=examples,
        inputs=paper_id,
        cache_examples=False,
        fn=get_paper_details,
        outputs=output,
    )

demo.launch()
