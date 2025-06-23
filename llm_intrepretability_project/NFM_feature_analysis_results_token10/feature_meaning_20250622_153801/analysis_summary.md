# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: checkpoints_topk/best_model.pt
Max Token Length: 10

## Analyzed Features

### Feature 21607

**Pattern:** Phrases starting with "that"

**Statistics:**
- Max activation: 28.0630
- Mean activation: 0.0202
- Percent active tokens (non-special): 0.57%

**Top Examples:**

Example 1:
Full text: ```
that year and into 1921 .
```
Activation context: ```
that year and into 1921
```
Max Token: 'that'
Activation: 3.1968

Example 2:
Full text: ```
that can be used to acquire and link different skills
```
Activation context: ```
that can be used to acquire and link different
```
Max Token: 'that'
Activation: 3.1181

Example 3:
Full text: ```
that Senjō no Valkyria
```
Activation context: ```
that Senjō no Valky
```
Max Token: 'that'
Activation: 3.1181

**Clamping Results:**

Generic prompt clamping:
Base: ```

Kate: I don't know. It's a big question.
Kate: You know, we have a lot of animals in the zoo.
Doesn't Kate know that the animals in the zoo aren't real.
I'd say I'm a cat person (cats are good looking, don't you think?) but I've always liked reptiles too.
I am not sure about that. Sometimes I like cats (they are very cute and I am sure they are nice pets), but I also like dogs.
The fact is that I like animals and I like people. I like both.
But you know what? I like you. I would like to know you.
I like you too. I would like to know you.
I have to go now. I will be back tomorrow.
You too. I will be back tomorrow.
I really like you and I want to know you.
I think you are a very nice person. Do you know why?
It's because I am also a nice person.
I'm not sure what you are talking about.
I'm not sure what you are talking about. I don't like you.
I think you are very nice and I will be back tomorrow.
I think you are very nice and I will be back tomorrow. I like you.
I think you are very nice and I
```
Clamped to 0.0: ```

Finch: Well, I don't know--I don't usually have any favorites.
Human: What do you think about when you're alone?
Finch: I think mostly about how much I'm going to miss you.
Human: What do you think about when you're alone?
Finch: I think mostly about how much I'm going to miss you. I think about it all the time.
Human: What do you think about when you're alone?
Finch: I think about how much I'm going to miss you. I think about it all the time.
Human: What do you think about when you're alone?
Finch: I think about how much I'm going to miss you. I think about it all the time.
Human: What do you think about when you're alone?
Finch: I think about how much I'm going to miss you. I think about it all the time.
Human: What do you think about when you're alone?
Finch: I think about how much I'm going to miss you. I think about it all the time.
Human: What do you think about when you're alone?
Finch: I think about how much I'm going to miss you. I think about it all the time.
Human: What do you think about when you're
```

Clamped to 56.1261: ```

It's a bird. It's a plane. It's a... dog?
As a dog owner, I'm sure that I'm not alone in my pet's preference for my company. And it's not just pets that are partial to their owners.
According to a recent study by the University of Texas at Austin, human-animal bonds are strong, and we don't even have to be family members for them to be strong.
According to the study, 48 percent of people who live with a dog or cat reported a very strong or strong bond with them, compared to 37 percent of people who live with a human loved one.
I'm not surprised by the results, but I think that they're a bit skewed.
The study, which surveyed 2,411 people, also found that people who live with a pet are more likely to be married, and 50 percent of couples reported that their relationship with their pet is a major part of their relationship.
The study also determined that people who live alone were more likely to own a pet than people who live with a partner.
And they're not just more likely to own a pet, but they're also more likely to have a higher risk of heart disease, stroke, and cancer.
The most surprising finding in the study was that people who own a pet are more likely than people
```

Clamped to 140.3152: ```

Chloe: Dog.
Human: Why?
Chloe: I can't have one.
Human: Why?
Chloe: Because I'm allergic to dog hair.
Human: Can you have a cat?
Chloe: No.
Human: Why?
Chloe: Because I'm allergic to cat hair.
Human: Can you have a bird?
Chloe: No. I'm allergic to bird poop.
Human: Can you have a horse?
Chloe: No. I'm allergic to horse hair.
Human: Can you have a cow?
Chloe: No. I'm allergic to cows' milk.
Human: Why?
Chloe: Because I'm lactose-intolerant.
Human: Can you have a sheep?
Chloe: No.
Human: Can you have a pig?
Human: Can you have a chicken?
Human: Can you have a bat?
Chloe: No. I'm allergic to bat poop.
Human: Can you have a duck?
Human: Can you have a duck egg?
Human: Can you have a rabbit?
Human: Can you have a turtle?
Human: Can you have a fish?
Human: Can you have a fish egg?
Human: Can you have a mollusk?
Human: Can you have a snail?

```

Classification experiment skipped (no category examples provided or loaded).

---

