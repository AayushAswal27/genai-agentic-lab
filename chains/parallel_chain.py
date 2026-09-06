from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel   # video uses langchain.schema 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto",
)

model1 = ChatHuggingFace(llm=llm) 


model2 = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Support Vector Machines (SVMs) are supervised machine learning algorithms
that can be used for classification, regression, and outlier detection.
The main idea behind an SVM classifier is to find the best decision boundary
that separates different classes of data. This decision boundary is called
a hyperplane. The SVM tries to choose the hyperplane that has the maximum
possible distance from the closest data points of each class. This distance
is called the margin. The data points that are closest to the decision
boundary are called support vectors because they determine the position
and orientation of the decision boundary.

For example, suppose we have a dataset containing information about houses
and we want to classify them as either affordable or expensive. The SVM
algorithm can use features such as house size, number of bedrooms, location,
and age of the house to find a boundary that separates the two categories.
If the data is linearly separable, a linear SVM can find a straight line in
two dimensions, or a hyperplane in higher dimensions, that separates the
classes. However, real-world datasets are often not linearly separable.
In such cases, SVMs can use kernel functions to transform the data into a
higher-dimensional space where a separating boundary may be easier to find.

One of the most commonly used kernels is the radial basis function (RBF)
kernel. The RBF kernel can create nonlinear decision boundaries and is useful
when the relationship between the features and the target variable is
complex. Other commonly used kernels include the linear, polynomial, and
sigmoid kernels. The choice of kernel can have a significant effect on the
performance of the model, so it is often selected through experimentation
or hyperparameter tuning.

SVMs also have important hyperparameters such as C and gamma. The C
parameter controls how strongly the model penalizes incorrectly classified
training examples. A large value of C tries to classify the training data
more strictly, while a smaller value allows some misclassification in
exchange for a wider margin. The gamma parameter is mainly relevant for
nonlinear kernels such as RBF. It controls how much influence an individual
training example has on the decision boundary. A high gamma value can lead
to a more complex decision boundary, while a low gamma value generally
produces a smoother boundary.

SVMs can work very well when the number of features is large and the dataset
is relatively small or medium-sized. However, they can become computationally
expensive on very large datasets, especially when nonlinear kernels are used.
Feature scaling is also important for SVMs because the algorithm is sensitive
to the scale of the input features. Standardization or normalization is
therefore commonly performed before training an SVM.

Overall, Support Vector Machines are powerful and flexible machine learning
algorithms. They are particularly useful for classification problems where
there is a clear separation between classes, but their performance depends
on selecting suitable hyperparameters, choosing an appropriate kernel, and
properly preprocessing the input data.
""" 

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()