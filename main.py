import evaluate
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from constants import prompts, prompts_inversia, prompts_anafora, prompts_ellipsis, prompts_add, prompts_passiv

def generation_microsoft(prompts, mode):
    '''
    Function that generates answers on all prompts from the list
    :param prompts: a list of prompts
    :param mode: mode for generation 0-English, 1-Russian, 2-Chinese
    :return: list of answers
    '''
    global pipe
    modes = ['You are a helpful AI assistant.', 'Ты полезный ИИ-помощник.', '你是一个有用AI助理。' ]
    # Подготовка всех промптов
    all_prompts = [
        [{"role": "system", "content": f"{modes[mode]}"}, {"role": "user", "content": f'{item}'}]
        for item in prompts
    ]

    # Генерация ответов батчами
    generation_args = {
        "max_new_tokens": 150,
        "do_sample": False,
    }
    answers = pipe(all_prompts, **generation_args)

    # Извлечение ответов
    answers = [result[0]['generated_text'] for result in answers]
    return [next(msg["content"] for msg in dialog if msg["role"] == "assistant") for dialog in answers]


# Функция для вычисления BERTScore
def get_bert_scores(prompts, answers):
    global bertscore
    results = bertscore.compute(
        predictions=answers,
        references=prompts,  # Сравниваем ответы с промптами
        model_type="distilbert-base-uncased"
    )
    return results["f1"]

def show_bert_diagram(scores1, scores2, n, lang):
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width/2, scores1, width, label='Ответы на обычные промты', color='skyblue')
    rects2 = ax.bar(x + width/2, scores2, width, label='Ответы на измененные', color='salmon')

    ax.set_ylabel('BERTScore')
    ax.set_title(f'Сравнение ответов по BERTScore для каждого промпта({lang})')
    ax.set_xticks(x)
    ax.set_xticklabels([i + 1 for i in range(n)], rotation=45, ha='right')
    ax.legend()

    # Добавление значений на столбцы
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()


def show_statistics(dataset1, answers1, dataset2, answers2, lang):
    scores1 = get_bert_scores(dataset1, answers1)
    scores2 = get_bert_scores(dataset2, answers2)

    k1 = 0 #счетчик для случаев, когда до изменения промта было лучше
    k2 = 0 #счетчик для случаев, когда после изменения промта стало лучше
    for i, (score1, score2) in enumerate(zip(scores1, scores2)):
        print(f"Prompt {i+1}: {score1:.2f}", f"{score2:.2f}")
        if score1 > score2:
            k1 += 1
        if score1 < score2:
            k2 += 1
    print(k1, k2)

    show_bert_diagram(scores1, scores2, len(scores1), lang)

def main():
    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=8
    )

    bertscore = evaluate.load("bertscore")

    # Промты обычные
    prompts_rus = [i[1] for i in prompts]
    answers_rus = generation_microsoft(prompts_rus, 1)

    prompts_ch = [i[2] for i in prompts]
    answers_ch = generation_microsoft(prompts_ch, 2)

    prompts_en = [i[0] for i in prompts]
    answers_en = generation_microsoft(prompts_en, 0)


    # Промты с применением пассива
    prompts_pass_rus = [i[1] for i in prompts_passiv]
    answers_pass_rus = generation_microsoft(prompts_pass_rus, 1)
    show_statistics(prompts_rus, answers_rus, prompts_pass_rus, answers_pass_rus, 'RUS')

    prompts_pass_ch = [i[2] for i in prompts_passiv]
    answers_pass_ch = generation_microsoft(prompts_pass_ch, 2)
    show_statistics(prompts_ch, answers_ch, prompts_pass_ch, answers_pass_ch, 'CH')

    prompts_pass_en = [i[0] for i in prompts_passiv]
    answers_pass_en = generation_microsoft(prompts_pass_en, 0)
    show_statistics(prompts_en, answers_en, prompts_pass_en, answers_pass_en, 'EN')

    # Промты со вставными конструкциями
    prompts_add_rus = [i[1] for i in prompts_add]
    answers_add_rus = generation_microsoft(prompts_add_rus, 1)
    show_statistics(prompts_rus, answers_rus, prompts_add_rus, answers_add_rus, 'RUS')

    prompts_add_ch = [i[2] for i in prompts_add]
    answers_add_ch = generation_microsoft(prompts_add_ch, 2)
    show_statistics(prompts_ch, answers_ch, prompts_add_ch, answers_add_ch, 'CH')

    prompts_add_en = [i[0] for i in prompts_add]
    answers_add_en = generation_microsoft(prompts_add_en, 0)
    show_statistics(prompts_en, answers_en, prompts_add_en, answers_add_en, 'EN')


    # Промты с анафорой
    prompts_anafora_rus = [i[1] for i in prompts_anafora]
    answers_anafora_rus = generation_microsoft(prompts_anafora_rus, 1)
    show_statistics(prompts_rus, answers_rus, prompts_anafora_rus, answers_anafora_rus, 'RUS')

    prompts_anafora_ch = [i[2] for i in prompts_anafora]
    answers_anafora_ch = generation_microsoft(prompts_anafora_ch, 2)
    show_statistics(prompts_ch, answers_ch, prompts_anafora_ch, answers_anafora_ch, 'CH')


    prompts_anafora_en = [i[0] for i in prompts_anafora]
    answers_anafora_en = generation_microsoft(prompts_anafora_en, 0)
    show_statistics(prompts_en, answers_en, prompts_anafora_en, answers_anafora_en, 'EN')


    # Промты с эллипсисом
    prompts_ellipsis_rus = [i[1] for i in prompts_ellipsis]
    answers_ellipsis_rus = generation_microsoft(prompts_ellipsis_rus, 1)
    show_statistics(prompts_rus, answers_rus, prompts_ellipsis_rus, answers_ellipsis_rus, 'RUS')


    prompts_ellipsis_ch = [i[2] for i in prompts_ellipsis]
    answers_ellipsis_ch = generation_microsoft(prompts_ellipsis_ch, 2)
    show_statistics(prompts_ch, answers_ch, prompts_ellipsis_ch, answers_ellipsis_ch, 'CH')


    prompts_ellipsis_en = [i[0] for i in prompts_ellipsis]
    answers_ellipsis_en = generation_microsoft(prompts_ellipsis_en, 0)
    show_statistics(prompts_en, answers_en, prompts_ellipsis_en, answers_ellipsis_en, 'EN')


    # Промты с инверсией
    prompts_inversia_rus = [i[1] for i in prompts_inversia]
    answers_inversia_rus = generation_microsoft(prompts_inversia_rus, 1)
    show_statistics(prompts_rus, answers_rus, prompts_inversia_rus, answers_inversia_rus, 'RUS')

    prompts_inversia_ch = [i[2] for i in prompts_inversia]
    answers_inversia_ch = generation_microsoft(prompts_inversia_ch, 2)
    show_statistics(prompts_ch, answers_ch, prompts_inversia_ch, answers_inversia_ch, 'CH')


    prompts_inversia_en = [i[0] for i in prompts_inversia]
    answers_inversia_en = generation_microsoft(prompts_inversia_en, 0)
    show_statistics(prompts_en, answers_en, prompts_inversia_en, answers_inversia_en, 'EN')



if __name__ == "__main__":
    main()