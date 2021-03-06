var data = 
{
  "start": {
    "start": "page",
    "stage": "chapter_1_start",
    "end": "paragraph"
  },
  "chapter_1_start": {
    "start": "page",
    "stage": "commenda"
  },
  "commenda": {
    "start": "paragraph",
    "stage": "start_commenda_talk",
    "end": "paragraph"
  },
  "start_commenda_talk": {
    "start": "page",
    "stage": "commenda_talk_hi"
  },
  "commenda_talk_hi": {
    "start": "paragraph",
    "stage": "commenda_no_money"
  },
  "commenda_no_money": {
    "start": "paragraph",
    "stage": "sho_delat", 
    "end": "paragraph" 
  },
  "sho_delat": {
    "start": "page",
    "stage": "commenda_second_attempt",
    "end": "paragraph"
  },
  "commenda_second_attempt": {
    "start": "page",
    "stage": "commenda_conditions",
    "end": "paragraph"
  },
  "commenda_conditions": {
    "start": "page",
    "stage": "commenda_vnuchok",
    "end": "paragraph"
  },
  "commenda_vnuchok": {
    "start": "page",
    "stage": "end",
    "end": "paragraph"
  },
  "end": {
    "text": "Кінець першої частини"
  }
};

var text_stage = {
  "start": {
    "ukr": "<h2>Obschaga.LIVE</h2>Привіт, новобранцю. Літо закінчується і ти " + 
           "вирушаєш на навчання до університету. Але перед тим, як почнуться заняття, " + 
           "тобі потрібно пройти ще один етап дорослого життя - гуртожиток. КПІшний гуртожиток.",
    "en": "wef"
  },
  "chapter_1_start": {
    "ukr": "<h2>Частина перша. Коменда.</h2>Кароч ти приїжджаєш і розумієш, що " + 
           "хочеш жити не в тому гуртожитку, до якого тебе записали. По суті так нізя. " + 
           "Але ти не збираєшся здаватися.",
    "en": "lkf"
  },
  "commenda": {
    "ukr": "Коменда - персона строга, але має свої вразливі місця. Подейкують, що " + 
           "за студентів, які готові відпрацьовувати, вона здатна продати душу дияволу. " + 
           "Як домовитися? Вибір за тобою. Головне пам'ятати про основну мету - поселитися в цей гуртожиток.",
    "en": "wlekf"
  },
  "start_commenda_talk": {
    "ukr": "<h4>В кабінеті коменди.</h4> Ти переступив поріг небезпечної зони. " + 
           "Шляху назад немає. Будь пусічкою і виберешся живим.",
    "en": "efw"
  },
  "commenda_talk_hi": {
    "ukr": "- Доброго дня! З питань поселення в гуртожиток звертатися до мене. Але, " + 
           "нажаль, не зможу вам нічим допомогти, адже списки на поселення же сформовані " + 
           "і редагуються лише  екстренних випадках.",
    "en": "efw"
  },
  "commenda_no_money": {
    "ukr": "- Ви повинні зрозуміти, що це незаконно і порушує правила проживання в " + 
           "студмістечку. Мене можуть звільнити. ",
    "en": "efw"
  },
  "sho_delat": {
    "ukr": "<h4>Хотілося б сказати, що ти лошара, але я є лише плодом твоєї уяви.</h4>" + 
           "<p>Ти подивився на умови проживання в гуртожитку, до якого був записаний " + 
           "з самого початку і зрозумів, що внатурі діло не альо.</p><p>Разом з тим, " + 
           "коменда кращого гуртожитку тебе послала. Але ж ти прекрасно знаєш, що вона " + 
           "може вирішити твою проблему. Висновок: погано просиш.</p><p>Кароч, солнце, " + 
           "давай-но збирайся з силоньками, і рули назад в кабінет до цієї жінки. Застосуй " + 
           "всі свої таланти і переконай її, що з твоєю появою її карма очиститься від " + 
           "усіх гріхів, а боженька пробачить їй всі знущання і крики на студентів про оплату. " + 
           "Бери харизмою і знай: якщо ти не досягнеш мети - я піду від тебе.</p><i>(голос в твоїй голові)</i>"
  },
  "commenda_second_attempt": {
    "ukr": "<h4>В кабінеті коменди.</h4>Все та ж небезпечна зона. Будь обережний, " + 
           "новобранцю. Пахне... відпрацюваннями."
  },
  "commenda_conditions": {
    "ukr": "<p>Хоть ти, канєшно, і дурак, але комєнда почуяла, що з тебе можна щось " + 
           "збити і очі в неї загорілися, наче в дитинки маленької при вигляді нової " + 
           "цяцьки.</p><p>Вона кароч тіпа подумала, і запропонувала кілька варіантів.</p>" + 
           "<p>Отже, ти можеш зганяти в магаз і придбати певні засоби і продукти (\"канєша " + 
           "ж благотворітєльность\"), можеш окультурити і посадити клумбу біля входу в " + 
           "гуртожиток (зайнятися \"суспільно корисними ділами\"), або ж розібратися, " + 
           "чому повільно працює компуктер в цієї жінки (ти ж праграміст, шолі).</p>"
  },
  "commenda_vnuchok": {
    "ukr": "<p>Ну какби дєло в шляпє. Тебе селять в гуртожиток. Конгратюлейшн.</p><p>До речі, поки ти там возився, до комєнди " + 
           "прибіг внучок. Ох, яка ж це кончена дитина. І кароч, воно пригає, бігає, просить " + 
           "погратися з ним. В якийсь момент ти розумієш, що краще погодитися, можливо так він " + 
           "швидше втомиться і засне. Або звалить.</p>"
  }
};

var choice_description = {
  "start": [
    {
      "text": {
        "ukr": "Прийнято, вриваємось",
        "en": ""
      },
      "action": "player.energy = 100; player.respect = 100; player.cash = 1000, player.mood = 100"
    }
  ],
  "chapter_1_start": [
    {
      "text": {
        "ukr": "Го, рішаємо",
        "en": ""
      }
    }
  ],
  "commenda": [
    {
      "text": {
        "ukr": "Ок. Почати розмову з комендою.",
        "en": ""
      }
    }
  ],
  "start_commenda_talk": [
    {
      "text": {
        "ukr": "- Доброго дня, шановна. Я дуже хочу поселитися у вашому гуртожитку.",
        "en": ""
      },
      "prompt": {
        "ukr": "- Доброго дня, шановна. Я дуже хочу поселитися у вашому гуртожитку. " + 
               "Проте, так історично склалося, що записали мене до іншого гуртожитку. " + 
               "Скажіть, будь ласка, чи можете ви допомогти мені?",
        "en": ""
      },
      "action": "player.respect += 5;"
    }, 
    {
      "text": {
        "ukr": "- Вєчєр в хату! Хочу жити в цих хоромах.",
        "en": ""
      },
      "prompt": {
        "ukr": "- Драстє, дарагуша. Хочу жити в цих хоромах, пуша тут мій бро. Но " + 
               "записали мене нє туда. Памагітє, плєс.",
        "en": ""
      },
      "action": "player.respect -= 5;"
    }, 
    {
      "text": {
        "ukr": "- Йоу. Треба за поселення добазаритись.",
        "en": ""
      },
      "prompt": {
        "ukr": "- Йоу. Тут можна за поселення добазаритись?",
        "en": ""
      },
      "action": "player.respect -= 5;"
    }
  ],
  "commenda_talk_hi": [
      {
        "text": {
          "ukr": "- Так, я це чудово розумію.",
          "en": ""
        },
        "prompt": {
          "ukr": "- Так, я це чудово розумію. Але мені дуже подобається саме " + 
                 "ЦЕЙ гуртожиток і я буду дуже вдячний, якщо Ви допоможете мені з поселенням.",
          "en": ""
        },
        "action": "player.respect += 5;"
      }, 
      {
        "text": {
          "ukr": "- Та то понятно, йоу.",
          "en": ""
        },
        "prompt": {
          "ukr": "- Та то понятно, йоу. Давайте якось не через касу проблемку вирішувати.",
          "en": ""
        },
        "action": "player.respect -= 5;"
      }, 
      {
        "text": {
          "ukr": "- Давайте по ділу: яка ціна питання?",
          "en": ""
        },
        "prompt": {
          "ukr": "- Давайте по ділу: яка ціна питання?",
          "en": ""
        },
        "action": "player.respect -= 5;"
      }
  ],
  "commenda_no_money": [
    {
      "text": {
        "ukr": "- Акєй, пардонтє."
      },
      "prompt": {
        "ukr": "- Добре, дякую. Гарного дня."
      },
      "action": "player.mood -= 10;"
    }
  ],
  "sho_delat": [
    {
      "text": {
        "ukr": "- Та всьо, харе, іду, іду."
      }, 
      "action": "player.energy -= 5"
    }
  ],
  "commenda_second_attempt": [
    {
      "text": {
        "ukr": "Впасти на коліна і слізно просити поселити в гуртожиток."
      },
      "prompt": {
        "ukr": "<i>Ти вриваєшся в кабінет коменди і падаєш на коліна, починаєш " + 
               "слізно просити про поселення.</i><p>- Розумієте, я маю жити тут. В цьому " + 
               "гуртожитку живе мій кращий друг, він - моя єдина підтримка. Ми скільки " + 
               "разом пережили, що здається наче ми сіамські близнюки. Розділити нас може " + 
               "лише пляшка пива. Ви - не пляшка пива.</p>"
      },
      "action": "player.mood /= 2; player.respect -= 5;"
    }, 
    {
      "text": {
        "ukr": "Відкрити двері з ноги."
      },
      "prompt": {
        "ukr": "<i>З криком \"This is SPARTAAAA\" ти вриваєшся до кабінету коменди.</i>" + 
               "<p>- Женщіна, давайте домовлятися. Я знаю, що у вас жизнь, какби, не маліна. " + 
               "Так давайте нарішаю: канфєтки, шоколадки, благодійність, відпрацювання. " + 
               "Для вас - хоч зірку з неба.</p>"
      },
      "action": "player.respect -= 15; player.mood += 10; player.energy -= 5"
    }, 
    {
      "text": {
        "ukr": "\"Тіше єдєш - дальше будєш\". Ведеш себе скромно. "
      },
      "prompt": {
        "ukr": "<i>Чемно заходиш до кабінету, розумієш як убого виглядаєш збоку, тому " + 
               "поводишся скромно і тихенько.</i><p>- Ми сьогодні вже зустрічалися. Я, " + 
               "все ж, хочу повернутися до старої розмови. Мені необхідно, от просто " + 
               "життєво необіхно жити у вашому гуртожитку. Позяяязя, давайте спробуємо домовитися.</p>"
      },
      "action": "player.respect += 5; player.mood += 5;"
    }  
  ],
  "commenda_conditions": [    
    {
      "text": {
        "ukr": "Рулю в магаз."
      },
      "action": "player.mood *= 2; player.energy -= 2; player.cash -= 200;"
    }, 
    {
      "text": {
        "ukr": "Я трудяжка, го всьо помию."
      },
      "action": "player.respect -= 5; player.mood *= 2; player.energy -= 10"
    }, 
    {
      "text": {
        "ukr": "Чиню компудахтєр, мамкин програміст в ділі."
      },
      "action": "player.respect += 5; player.mood *= 2;"
    }  
  ],
  "commenda_vnuchok": [
    {
      "text": {
        "ukr": "Розважати малого."
      },
      "action": "location.replace(\"mini-games/hangman.html\");"
    }
  ]
};