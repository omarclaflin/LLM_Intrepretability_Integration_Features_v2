# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: checkpoints_topk/best_model.pt
Max Token Length: 10

## Analyzed Features

### Feature 21781

**Pattern:** Abstract concepts

**Statistics:**
- Max activation: 2.6236
- Mean activation: 0.0657
- Percent active tokens (non-special): 10.80%

**Top Examples:**

Example 1:
Full text: ```
the empire 's complex mix of religious traditions ,
```
Activation context: ```
the empire 's complex mix of religious traditions
```
Max Token: 'complex'
Activation: 0.7011

Example 2:
Full text: ```
morphological and molecular phylogenetic evidence .
```
Activation context: ```
morphological and molecular phylogenetic evidence
```
Max Token: 'ological'
Activation: 0.6975

Example 3:
Full text: ```
. The Egyptians ' visual representations of their gods
```
Activation context: ```
. The Egyptians ' visual representations of their
```
Max Token: 'visual'
Activation: 0.6587

**Clamping Results:**

Generic prompt clamping:
Base: ```

Kim: Bugs. I love bugs.
Human: What's your favorite food?
Kim: Peanut butter. I love peanut butter.
Human: What's your favorite color?
Kim: Dark blue. I like my skin to be dark blue. And I like my eyes to be dark blue.
Human: What's your favorite song?
Kim: "The Sound of Music."
Human: Where are you from?
Kim: I was born in Seoul, Korea.
Human: If you could be anything, what would you want to be?
Kim: A doctor because I like helping people.
Human: What does a doctor do?
Kim: They help people who are sick.
Human: What do you want to be when you grow up?
Kim: I want to be a doctor.
Human: What do you think a doctor does?
Kim: They help people who are sick.
Human: What do you like to do?
Kim: I like to play.
Human: What do you like to eat?
Human: What do you like to wear?
Kim: I like to wear a good-looking shirt and a nice pair of shoes.
Human: What colors do you like most?
Kim: Pink and blue. I like pink.
Kim: A lot of people like to eat.

```
Clamped to 0.0: ```

Personal: What do you want to be when you grow up?
Personal: If you had a million dollars, what would you spend it on?
Personal: Would you rather read a book or watch TV?
Personal: Do you want to study in college?
Personal: Do you want to have a career?
Personal: How often do you play sports?
Personal: What do you want to do next summer?
Personal: What do you want to do when you grow up?
Personal: Do you want to learn English?
Personal: How many times have you been to the hospital?
Personal: What is your favorite music?
```

Clamped to 5.2472: ```

Human: What's your name?
Rabbit: Hi, my name is Rabbit.
Human: What's your favorite animal?
Rabbit: I like to go to the park and feed the squirrels.
Human: What's your favorite animal?
Rabbit: I like to go to the park and feed the squirrels.
Human: What's your favorite animal?
Rabbit: I like to go to the park and feed the squirrels.
Human: What's your favorite animal? What's your favorite animal?
Rabbit: I like to go to the park and feed the squirrels.
Human: What's your favorite animal? What's your favorite animal? What's your favorite animal?
Rabbit: I like to go to the park and feed the squirrels. What's your favorite animal? What's your favorite animal? What's your favorite animal?
Human: What's your favorite animal?
Rabbit: I like to go to the park and feed the squirrels.
Human: What's your favorite animal? What's your favorite animal? What's your favorite animal? What's your favorite animal?
Human: What's your favorite animal?
Rabbit: I like to go to the park and feed the squirrels. What's
```

Clamped to 13.1181: ```

Human: What's your favorite animal?
Humans: The dog was the first animal.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The dog was the first animal that we domesticated.
Humans: The
```

Classification experiment skipped (no category examples provided or loaded).

---

