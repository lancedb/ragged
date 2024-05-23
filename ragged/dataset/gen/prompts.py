Q_FROM_CONTEXT_DEFAULT = \
""""
Your task is to write a list of {num_questions} detailed statements or questions for given a context. It should satisfy the following conditions:
* should relevant to the specifc context. and should not be a one-liner. Rahter it should be a detailed question or statement.
* MUST not be keyword based. Try to make it semantically similar to the context without using the same words. Basically string matching should not work for searching the answer.
* MUST NOT mention something like "according to the passage" or "context".

Provide your answer as a list named 'content'
Now here is the context.
Context: {context}\n
"""

QA_FROM_CONTEXT_DEFAULT = \
""""
Your task is to write a list of {num_questions} detailed statements or questions & their answers for given a context. It should satisfy the following conditions:
* should relevant to the specifc context. and should not be a one-liner. Rahter it should be a detailed question or statement.
* MUST not be keyword based. Try to make it semantically similar to the context without using the same words. Basically string matching should not work for searching the answer.
* MUST NOT mention something like "according to the passage" or "context".

Provide your questions and answers as a list named 'content' with each element being a dictionary with keys '{question}' and '{answer}'.
Now here is the context.
Context: {context}\n
"""