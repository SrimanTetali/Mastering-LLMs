# **Day 1[Part 2]: Domain and Task Adaption Methods**

   How to use LLMs Effectively

   Types of Domain Adaptation Methods

   Domain-Specific Pre-Training

   Domain-Specific Fine-Tuning

   Retrieval Augmented Generation (RAG)

   Choosing Between RAG, Domain-Specific Fine-Tuning, and
    Domain-Specific Pre-Training

# **How to use LLMs Effectively:**

When we talk about using AI models like ChatGPT effectively, we need to
understand that while these models are great at generating text on many
topics, they sometimes struggle with specific areas. We call these areas
\"domains.\" For example, in healthcare, there are specific terms and
concepts that are really important, but a general AI might not fully
understand them.

That\'s where domain-specific AI models come in. These are like
specialized versions of ChatGPT that are trained to understand specific
fields, like healthcare or finance. They\'re better at grasping the
details and nuances of these areas.

benefits of using domain-specific LLMs:

1.   **They\'re More Precise:**  General AI models might be good at
    talking about lots of things, but domain-specific ones really know
    their stuff. They understand the special words and phrases used in a
    particular field, making their answers more accurate.

2.  **They Avoid Mistakes:** General AI models can sometimes make
    mistakes or say things that don\'t make sense in a specific context.
    But domain-specific models are less likely to do this because
    they\'re trained to understand the ins and outs of their field.

3.  **Better User Experience:** When you\'re talking to an AI, you want
    it to understand you and give you helpful answers. Domain-specific
    models are better at this because they know the specific details of
    the topic you\'re asking about.

4.  **They Make Things Faster:** Using domain-specific AI models can
    help businesses be more efficient. They can do tasks automatically
    and use the right words for the job, saving time and allowing people
    to focus on more important work.

5.  **They Keep Data Safe:** In industries like healthcare, privacy is
    really important. Domain-specific AI models are designed to keep
    sensitive information secure and follow all the rules about privacy.

# **Types of Domain Adaptation Methods:**

There are various ways to teach Language Models (LLMs) about specific
domains or fields of knowledge. Each of these methods has its own
strengths and weaknesses.

Three of these methods are

  ## 1. **Domain-Specific Pre-Training:**

-   **Training Duration**: This method involves pre-training Large
    Language Models (LLMs) on extensive datasets, which can take days to
    weeks to months depending on the size of the dataset and
    computational resources.

-   **Summary**: Requires a large amount of domain training data; can
    customize model architecture, size, tokenizer, etc.

> **Explanation:** LLMs like PaLM 540B, GPT-3, and LLaMA 2 are trained
> on huge datasets with billions to trillions of words. They cover all
> sorts of language topics. For example, special training makes models
> like ESMFold and ProGen2 for proteins, Galactica for science,
> BloombergGPT for finance, and StarCoder for code. These models usually
> work better than general ones in their areas. But sometimes, they can
> still make mistakes because the topics are so complex.

  ## 2. **Domain-Specific Fine-Tuning**:

-   **Training Duration**: Fine-tuning typically takes minutes to hours.

-   **Summary**: It involves training a pre-trained LLM on specific
    tasks or domains, thereby adapting its knowledge to a narrower
    context.

> **Explanation**: Fine-tuning is like customizing a pre-trained AI for
> specific jobs. For example, Alpaca, xFinance, and ChatDoctor are
> models that were fine-tuned for general, financial, and medical chat
> tasks, respectively. It\'s cheaper and faster than training from
> scratch, so it\'s a smart way to adapt the AI to different tasks.

  ## 3. **Retrieval Augmented Generation (RAG)**:

-   **Training Duration**: No specific training duration is required.

-   **Summary**: RAG integrates external or non-parametric knowledge
    into LLMs using an information retrieval system, without directly
    modifying the LLM\'s model weights.

> **Explanation**: In RAG, we give our AI extra info from an outside
> source to help it understand better. This method has some cool
> benefits: it doesn\'t need extra training, so it saves time and anyone
> can use it without needing to be an expert. Plus, we can show where
> the extra info comes from, which is handy. RAG helps the AI avoid
> making weird mistakes and lets us tweak the knowledge easily without
> messing up the AI\'s main training. Scientists are still working on
> making RAG even better by finding the best ways to mix outside info
> with what the AI already knows.

# **Domain-Specific Pre-Training**

![alt text](assets/domain_specific%20pre%20training.png)

Domain-specific pre-training involves training large language models
(LLMs) on extensive datasets that specifically represent the language
and characteristics of a particular domain or field. This process aims
to enhance the model\'s understanding and performance within a defined
subject area.

### **Example of BloombergGPT**:

BloombergGPT is a 50 billion parameter language model designed to excel
in various tasks within the financial industry. While general models are
versatile and perform well across diverse tasks, they may not outperform
domain-specific models in specialized areas. At Bloomberg, where a
significant majority of applications are within the financial domain,
there is a need for a model that excels in financial tasks while
maintaining competitive performance on general benchmarks.

### **Tasks Performed by BloombergGPT**:

1.  **Financial Sentiment Analysis**: BloombergGPT reads financial texts
    like news articles or social media posts and figures out if people
    are feeling positive or negative about the market. This helps
    investors make smarter decisions.

2.  **Named Entity Recognition**: BloombergGPT looks at financial
    documents and recognizes important things like companies, people, or
    financial tools mentioned in them. This is super helpful for finding
    the right information in big piles of data.

3.  **News Classification**: It puts financial news articles into groups
    based on what they\'re about. This makes it easier for people to
    know what\'s happening in different parts of the financial world.

4.  **Question Answering in Finance**: If you have questions about the
    market or financial stuff, you can ask BloombergGPT, and it will
    give you useful answers.

5.  **Conversational Systems for Finance**: BloombergGPT can have
    conversations about finance with people. You can ask it questions,
    clear up confusion, or chat about financial topics. It\'s like
    talking to a smart friend who knows a lot about money.

### **How is BloombergGPT Trained?**:

To achieve its capabilities, BloombergGPT undergoes domain-specific
pre-training using a large dataset called FinPile. This dataset combines
domain-specific financial language documents from Bloomberg\'s extensive
archives with public datasets. FinPile consists of diverse English
financial documents, including news, filings, press releases,
web-scraped financial documents, and social media content. The training
corpus is roughly divided into half domain-specific text and half
general-purpose text, leveraging the advantages of both domain-specific
and general data sources.

# **Domain-Specific Fine-Tuning**:

Domain-specific fine-tuning is a process where we make a pre-existing
language model even better for a specific task or area. We do this to
improve its performance and make it work well in that particular
context. Instead of starting from scratch, we take a general language
model that\'s already been trained on lots of different language stuff
(like grammar and understanding words in different contexts) and
fine-tune it to focus on a narrower set of data related to a specific
domain or task.

## **Key Steps**:

1.  **Pre-training**: First, we train a big language model on a wide
    range of language examples, so it understands language patterns well
    (this is our general language model).

2.  **Fine-tuning Dataset**: Then, we collect or prepare a smaller
    dataset that\'s all about the specific domain or task we want the
    model to be good at. This dataset has examples and instances related
    to what we\'re focusing on.

3.  **Fine-tuning Process**: Next, we train our general language model
    further using this specific dataset. During this fine-tuning, we
    adjust the model\'s settings based on the new data, while still
    keeping the general language understanding it gained during
    pre-training.

4.  **Task Optimization**: Finally, we tweak the fine-tuned model to
    make it work even better for the specific tasks in that domain. This
    might mean adjusting things like the model\'s size or how it
    processes words to get the best performance.

## **Advantages**:

-   **Specialization**: It makes the model really good at a specific
    domain or task, so it\'s more effective for those kinds of jobs.

-   **Efficiency**: It saves time and resources compared to training a
    whole new model from scratch because we\'re building on what the
    model already knows.

-   **Improving Performance**: It helps the model understand the unique
    needs and details of the target domain better, so it performs better
    on tasks specific to that domain.

## **Example: ChatDoctor LLM**:

One popular example of domain-specific fine-tuning is the ChatDoctor
language model. It\'s a specialized model that\'s fine-tuned on
Meta-AI\'s large language model meta-AI (LLaMA) using a dataset of
patient-doctor dialogues from an online medical consultation platform.
This fine-tuning on real-world patient interactions makes ChatDoctor
really good at understanding patient needs and giving accurate medical
advice. It also uses information from sources like Wikipedia and medical
databases to make its responses even more accurate. ChatDoctor\'s
contributions include a method for fine-tuning language models in the
medical field, a dataset it\'s trained on, and a model that can keep
learning and updating itself with new medical knowledge.

# **Retrieval Augmented Generation (RAG)**:

Retrieval Augmented Generation (RAG) is a smart way to make the
responses from language models (LLMs) even better. It works by adding
fresh and relevant information from outside sources while the LLM is
generating its response. This helps to fill in any gaps in the LLM\'s
knowledge and reduces the chances of it giving wrong answers. RAG has
two main parts: first, it looks for and brings in the extra info, then
it uses this info along with what it already knows to come up with a
good answer.

## **How RAG Works**:

RAG involves two main steps: retrieval and content generation. In the
retrieval step, it looks for the most relevant info based on the query.
Then, in the content generation step, it combines this info with its own
knowledge to give the best possible response. This helps to make the
responses more accurate and trustworthy.

![alt text](assets/RAG_working.png)

## **Key Components of RAG**:

1.  **Ingestion**: This step breaks down documents into smaller pieces
    and creates special codes (embeddings) for each piece. These pieces
    are stored in a database, making it easier to find the right info
    later.

2.  **Retrieval**: Using the codes from the ingestion step, RAG looks
    through the database to find the top documents that match the query.

3.  **Synthesis**: Finally, RAG uses the info it found to craft a good
    response. It uses both the retrieved info and what it already knows
    to make sure the answer is spot on.

RAG comes with specific advantages and disadvantages. The decision to
employ or refrain from using RAG depends on an evaluation of these
factors.

## **Advantages of RAG** 
- ** Keeps Information Fresh:** RAG gets the latest or specific data from outside sources, making sure the AI\'s answers are up-to-date.
- ** Adds Domain-Specific Knowledge:** RAG helps the AI by giving it specialized information from a database tailored to a specific field.
- ** Reduces Mistakes and Shows Sources:** RAG makes it less likely for the AI to make mistakes by giving it reliable facts from external sources. It can also show where the information came from.
- ** Saves Money:** RAG is a cheaper option because it doesn\'t need as much training or fine-tuning as other methods.

## **Disadvantages of RAG**
- ** Complex Setup:** Setting up RAG can be complicated as it involves many different parts like databases and search systems. If any of these parts don\'t work well, RAG might not perform properly.
- ** Takes Longer to Respond:** Because RAG has to search through databases,it might take more time to come up with an answer compared to models that don\'t need to look outside.
     
# **Choosing Between RAG, Domain-Specific Fine-Tuning, and Domain-Specific
Pre-Training**

![alt text](assets/choosing%20types_domain_task.png)

+-------------------+-------------------------+------------------------+
| ** Use            | ** Use Domain-Specific  | ** Use RAG When:**     |
| Domain-Specific   | Fine-Tuning When:**     |                        |
| Pre-Training      |                         |                        |
| When:**           |                         |                        |
+===================+=========================+========================+
| ** Exclusive      | ** Specialization       | ** Information         |
| Domain Focus:**   | Needed:** When you      | Freshness Matters:**   |
|                   | already have a          |                        |
| When you need a   | pre-trained LLM and     | When you need          |
| model exclusively | want to adapt it for    | up-to-date,            |
| trained on data   | specific tasks or       | context-specific data  |
| from a specific   | within a particular     | from external sources. |
| domain.           | domain.                 |                        |
+-------------------+-------------------------+------------------------+
| ** Customizing    | ** Task Optimization:** | ** Reducing            |
| Model             |                         | Hallucination is       |
| Architecture:**   | To adjust the model\'s  | Crucial:** To ground   |
|                   | parameters related to   | LLMs with verifiable   |
| To customize      | the task for optimal    | facts and citations    |
| various aspects   | performance in the      | from an external       |
| of the model      | chosen domain.          | knowledge base.        |
| architecture,     |                         |                        |
| size, tokenizer,  |                         |                        |
| etc.              |                         |                        |
+-------------------+-------------------------+------------------------+
| ** Extensive      | ** Time and Resource    | ** Cost-Efficiency is a|
| Training Data     | Efficiency:**           | Priority:**            |
| Available:**      |                         |                        |
|                   | To save time and        | When avoiding          |
| When you have a   | computational resources | extensive model        |
| large amount of   | compared to training a  | training or            |
| domain-specific   | model from scratch.     | fine-tuning is         |
| training data     |                         | necessary, and you can |
| available.        |                         | implement without the  |
|                   |                         | need for training.     |
+-------------------+-------------------------+------------------------+

These tables provide a quick reference for deciding when to use each
approach based on specific requirements and priorities.
