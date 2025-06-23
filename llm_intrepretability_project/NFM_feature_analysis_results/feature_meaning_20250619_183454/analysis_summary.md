# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: checkpoints_topk/best_model.pt
Max Token Length: 10

## Analyzed Features

### Feature 17512

**Pattern:** Prepositional phrases

**Statistics:**
- Max activation: 0.8303
- Mean activation: 0.0027
- Percent active tokens (non-special): 0.53%

**Top Examples:**

Example 1:
Full text: ```
murder of the brother of the woman he loved .
```
Activation context: ```
murder of the brother of the woman he loved
```
Max Token: 'of'
Activation: 0.2429

Example 2:
Full text: ```
historical accuracy in favour of good storytelling . 

```
Activation context: ```
historical accuracy in favour of good storytelling .
```
Max Token: 'favour'
Activation: 0.2278

Example 3:
Full text: ```
in rural areas . 

```
Activation context: ```
in rural areas
```
Max Token: 'rural'
Activation: 0.2131

**Clamping Results:**

Generic prompt clamping:
Base: ```

Homestar Runner: That's one of the many mysteries of the universe.
Human: What are you afraid of?
Homestar Runner: That's one of the many mysteries of the universe.
Human: What are you afraid of?
Homestar Runner: That's one of the many mysteries of the universe. But I guess we'll just have to live with it.
This page was last edited on 12 August 2009, at 18:22.
```
Clamped to 0.0: ```

Human: Do you have any allergies or special health conditions?
Human: Do you have any particular food allergies or special health conditions?
Human: Do you have any special medical conditions?
Human: Do you have any special medical conditions?
Human: Do you have any chronic medical conditions or allergies?
Human: Do you have any chronic medical conditions or allergies?
Human: Do you have any chronic physical or mental health conditions?
Human: Do you have any chronic physical or mental health conditions?
Human: Do you have any chronic mental health conditions?
Human: Do you have any chronic mental health conditions?
Human: Do you have any chronic physical or mental health conditions?
Human: Do you have any chronic mental health conditions?
Human: Do you have any chronic physical or mental health conditions or allergies?
Human: Do you have any chronic physical or mental health conditions or allergies?
Human: Do you have any chronic health conditions or allergies?
Human: Do you have any chronic health conditions or allergies?
Human: Do you have any chronic physical or mental health conditions or allergies?
Human: Do you have any chronic physical or mental health conditions or allergies?
Human: Do you have any chronic physical or mental health conditions or allergies?
Human: Do you have any chronic physical or mental health conditions or allergies?
Human: Do you have any chronic physical or mental health
```

Clamped to 1.6605: ```

Human: What's the best thing about being human?
Human: What's the worst thing about being human?
Human: What's the most important skill to have growing up?
Human: What's the most important thing about being a human?
Human: What's the best thing about being a human?
Human: What's the worst thing about being a human?
Human: What's the most important skill to have growing up?
Human: What's the most important thing about being a human?
Human: What's the best thing about being a human?
Human: What's the worst thing about being a human?
Human: What's the most important skill to have growing up?
Human: What's the most important thing about being a human?
Human: What's the best thing about being a human?
Human: What's the worst thing about being a human?
Human: What's the most important skill to have growing up?
Human: What's the most important thing about being a human?
Human: What's the best thing about being a human?
Human: What's the worst thing about being a human?
Human: What's the most important skill to have growing up?
Human: What's the most important thing about being a human?
Human: What's the best thing about being a
```

Clamped to 4.1513: ```

Kid: Elephant
Human: What's your favorite color?
Kid: Pink
Human: What's your favorite dessert?
Kid: Cookies
Human: What is your favorite thing to do on the weekend?
Kid: Go to the park
Human: What is your favorite book?
Kid: Winnie the Pooh
Human: What is your favorite movie?
Kid: The Lion King
Human: What is your favorite holiday?
Kid: Easter
Human: What is your favorite sport?
Kid: Volleyball
Human: What is your favorite subject?
Kid: Math
Human: What is your favorite color?
Kid: Black
Human: What is your favorite place to go on vacation?
Kid: Disneyland!
Human: What is your favorite song?
Kid: Happy Birthday
Human: What is your favorite vegetable?
Kid: Lettuce
Human: What is your favorite fruit?
Kid: Banana
Human: What is your favorite animal?
Kid: Dog
Human: What is your favorite animal?
Kid: Cat
Human: What is your favorite food?
Kid: Pizza
Human: Why do you like playing video games?
Kid: I like to play with my friends.
Human: Why do you like playing with your friends?
Kid: It's fun.

```

Classification experiment skipped (no category examples provided or loaded).

---

