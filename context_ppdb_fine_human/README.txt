This data set consists of 3,750 term pairs, each given within a context sentence, built upon a subset of terms from PPDB 2.
Each term pair is annotated to the semantic relation that holds between the terms in the given contexts.

If you use this dataset, please cite the following paper:

Adding Context to Semantic Data-Driven Paraphrasing. Vered Shwartz and Ido Dagan. *SEM 2016.

File Structure: tab-separated text file

Fields:

- x: the first term

- y: the second term

- context_x: the sentence in which x appears (highlighted by <x>x</x>)

- context_y: the sentence in which y appears (highlighted by <y>y</y>)

- semantic_relation: the (directional) semantic relation that holds between x and y: equivalence, forward_entailment, reverse_entailment, alternation, other-related and independence.

- confidence: the relation annotation confidence (precentage of labelers that selected this relation), in a scale of 0-1
