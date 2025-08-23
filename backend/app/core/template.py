SENTIMENT_TEMPLATE = """
    Predict the sentiment of the following input sentence.
    The response must begin with "Sentiment: ", followed by one of these keywords: "positive", "neutral", "negative", to reflect the sentiment of the input sentence.

    Sentence: {input}
"""


MEDICAL_TEMPLATE = """
    Choose the correct option for the following question.

    ### Question:
    {question}

    ### Choices:
    A. {opa}.
    B. {opb}.
    C. {opc}.
    D. {opd}.

    ### Answer:
"""



