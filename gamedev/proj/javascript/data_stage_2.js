var data = 
{
  "start": {
    "start": "page",
    "stage": "chapter_2_start",
    "end": "paragraph"
  },
  "chapter_2_start": {
    "start": "page",
    "stage": "sosed_privet"
  },
  "sosed_privet": {
    "start": "paragraph",
    "stage": "nachinaem",
    "end": "paragraph"
  },
  "nachinaem": {
    "start": "page",
    "stage": "morning"
  },
  "morning": {
    "start": "page",
    "stage": "end"
  },
  "end": {
    "text": "Кінець другої частини"
  }
};

var text_stage = {
  "start": {
    "ukr": "<h2>Частина друга. В кімнаті.</h2> Ну чьо, захаді нє бойся, ухаді нє плачь.",
    "en": "wef"
  },
  "chapter_2_start": {
    "ukr": "Ти кароч зайшов і розумієш в яку с*аку ти попав. Ремонт не камільфо, але в " + 
           "кімнаті є металопластикове вікно. І пофіг, що тобі треба самому його вставити. " + 
           "Не все так погано, ти спостерігаєш наявність життя в цих розвалинах. Таргани:) ",
    "en": "lkf"
  },
  "sosed_privet": {
    "ukr": "Але... О Боже, що це? Тут є люди. Здається це твій сусід. З першого погляду він " + 
           "не дуже розумним здається, але що же зробиш, доведеться тобі з ним строк мотати. "
  }, 
  "nachinaem": {
    "ukr": "Сусід, з виду, наче й дурник, але тему шарить. Ще толком не познайомилися, а вже " + 
           "пропонує екскурсію на поляну. Тільки от не задача, тобі потрібно в кімнаті облаштуватися, " + 
           "хоча б ліжко своє знайти, чи що. Разом з тим, ти планував з падругою своєю зустрітися. " + 
           "І лише зараз ти розумієш всю складність буття: поляна = бухіч, прибирання = не побачишся з " + 
           "подругою, не побачишся з подругою = \"ой всьо\"."
  },
  "morning": {
    "ukr": "Утрєчко. Ну що, ти починаєш освоюватися, сьогодні День першокурсника і завтра тебе чекає нове " + 
           "випробування. Почнуться заняття і ти відчуєш всю гіркоту від матану, програмування на С та інших " + 
           "радощів життя. Перший раз завжди боляче, потім звикнеш, тобі навіть сподобається і ти будеш займатися " + 
           "цим щодня. <p>А зараз розчехляйся і займися чимось, бродяга.</p><p>Головне - не накидайся сьогодні так, " + 
           "щоб завтра померти.</p><i>Зошити можна будь-які. Навіть в кружечок. </i><i>Поки що це всі тексти, які ми " + 
           "залили ;) Перевір, наскільки добре ти вживешся з новими \"друзями\";)</i>"
  }
};

var choice_description = {
  "start": [
    {
      "text": {
        "ukr": "Захажу",
        "en": ""
      },
      "action": "player.energy = 100; player.respect = 100; player.cash = 1000, player.mood = 100"
    }
  ],
  "chapter_2_start": [
    {
      "text": {
        "ukr": "Хлоп тапком",
        "en": ""
      }
    }
  ],
  "sosed_privet": [
    {
      "text": {
        "ukr": "Прівєт, Вася. Ти - Вася, я - не Вася.",
        "en": ""
      }
    }
  ],
  "nachinaem": [
    {
      "text": {
        "ukr": "Го на півасік.",
        "en": ""
      },
      "prompt": {
        "ukr": "аєєєєєєєє, Ти побачив восьме чудо світу. Поляна. П-О-Л-Я-Н-А. Тебе катали у візочку з Сільпо, " + 
               "якраз по політеху йшла твоя дівчина, яку ти, нагадаю, морознув. Ах, так, морознув, бо типу прибирав "+ 
               "в кімнаті.Ну що ж, ти розвіявся, але завтра тебе чекає скандал, ще й в брудній кімнаті.",
        "en": ""
      },
      "action": "player.mood += 50; player.energy /= 2;"
    }, 
    {
      "text": {
        "ukr": "Текс, всталяємо вікна, ганяємо таракашок",
        "en": ""
      },
      "prompt": {
        "ukr": "В кімнаті чисто. Ура. Але ти без півасіка, ще й втомлений і падруга агриться. Тобі пофіг, ти йдеш спати.",
        "en": ""
      },
      "action": "player.mood -= 20; player.energy /= 2; player.respect -= 15;"
    }, 
    {
      "text": {
        "ukr": "Я каблук.",
        "en": ""
      },
      "prompt": {
        "ukr": "Ну кароч, пішов ти гуляти зі своєю ненаглядною, тобі не виносили мізки, все ок, тільки в кімнаті " + 
               "от срач. А ще тоя подруга жере наче в неї три шлунки, тому така прогулянка тобі вдарила по кишені.",
        "en": ""
      },
      "action": "player.mood += 20; player.respect += 15;"
    }
  ],
  "morning": [
    {
      "text": {
        "ukr": "Паєхалі",
        "en": ""
      },
      "action": "location.replace(\"mini-games/reflex.html\");"
    }
  ],
};