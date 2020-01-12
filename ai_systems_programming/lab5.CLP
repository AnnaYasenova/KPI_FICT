(defrule data­input
=>(printout t crlf "How much did the jevelry cost?")
(bind ?money (read))
(assert (money ?money))
(printout t crlf "Is it documentary licensed (1-yes, 0-no)")
(bind ?docs (read))
(assert (docs ?docs))
(printout t crlf "Does it have any blinks? (yes ­ 1/no ­ 0)")
(bind ?fur (read))
(assert (fur ?fur))
(printout t crlf "Enter the weight of thing in gramms?")
(bind ?size (read))
(assert (size ?size)))
(defrule R1
(fur ?fur)
(size ?size)
(test (= ?size 0))
=>
(printout t crlf crlf "Enter the weight of thing." crlf)
(assert (appearance "none"))
(assert (appearancecnst 0)))
(defrule R2
(fur ?fur)
(size ?size)
(test (< ?size 25))
=>
(printout t crlf crlf "Thing looks like precious." crlf)
(assert (appearance "likeprecious"))
(assert (appearancecnst 1)))
(defrule R3
(fur ?fur)
(size ?size)
(test (>= ?size 50))
=>
(printout t crlf crlf "Thing looks like precious." crlf)
(assert (appearance "likeprecious"))
(assert (appearancecnst 1)))
(defrule R4
(fur ?fur)
(size ?size)
(test (= ?fur 1))
=>
(printout t crlf crlf "Thing looks like precious." crlf)
(assert (appearance "likeprecious"))
(assert (appearancecnst 1)))
(defrule R5
(fur ?fur)
(size ?size)
(test (and(and(>  ?size 25) (<= ?size 50))(= ?fur 0)))
=>
(printout t crlf crlf "Thing doesn't look like precious." crlf)
(assert (appearance "notlikeprecious"))
(assert (appearancecnst 2)))
(defrule R6
(money ?money)
(docs ?docs)
(test (and(>  ?money 200) (= ?docs 1)))
=>
(printout t crlf crlf "This thing is reasonably priced." crlf)
(assert (moneycost "good"))
(assert (moneycostcnst 1)))
(defrule R7
(money ?money)
(docs ?docs)
(test (and(>  ?money 200) (= ?docs 0)))
=>
(printout t crlf crlf "The price is too high." crlf)
(assert (moneycost "bad"))
(assert (moneycostcnst 2)))
(defrule R8
(money ?money)
(docs ?docs)
(test (and(<  ?money 200) (= ?docs 0)))
=>
(printout t crlf crlf "This thing is reasonably priced, as thing without licence." crlf)
(assert (moneycost "normal"))
(assert (moneycostcnst 4)))
(defrule R9
(money ?money)
(docs ?docs)
(test (and(<  ?money 200) (= ?docs 1)))
=>
(printout t crlf crlf "The price is too low." crlf)
(assert (moneycost "notnormal"))
(assert (moneycostcnst 3)))
(defrule R10
(appearancecnst ?appearancecnst)
(moneycostcnst ?moneycostcnst)
(test (and(= ?appearancecnst 1) (= ?moneycostcnst 1)))
=>
(printout t crlf crlf "Thing is really precious." crlf)
(assert (act "glearn")))
(defrule R11
(appearancecnst ?appearancecnst)
(moneycostcnst ?moneycostcnst)
(test (and(= ?appearancecnst 1) (= ?moneycostcnst 0)))
=>
(printout t crlf crlf "Check the documents for this jewelery. It can be fake." crlf)
(assert (act "glearn")))
(defrule R12
(appearancecnst ?appearancecnst)
(moneycostcnst ?moneycostcnst)
(test (and(= ?appearancecnst 1) (= ?moneycostcnst 3)))
=>
(printout t crlf crlf "This thing is probable non presious, but cheap and looks good." crlf)
(assert (act "glearn")))
(defrule R13
(appearancecnst ?appearancecnst)
(moneycostcnst ?moneycostcnst)
(test (and(= ?appearancecnst 1) (= ?moneycostcnst 4)))
=>
(printout t crlf crlf "Check the documents for this jewelry. It can be fake." crlf)
(assert (act "glearn")))
(defrule R14
(appearancecnst ?appearancecnst)
(moneycostcnst ?moneycostcnst)
(test (and(= ?appearancecnst 0) (= ?moneycostcnst 1)))
=>
(printout t crlf crlf "Thing is really precious." crlf)
(assert (act "glearn")))
(defrule R15
(appearancecnst ?appearancecnst)
(moneycostcnst ?moneycostcnst)
(test (and(= ?appearancecnst 0) (= ?moneycostcnst 2)))
=>
(printout t crlf crlf "Thing isn't precious." crlf)
(assert (act "glearn")))
(defrule R16
(appearancecnst ?appearancecnst)
(moneycostcnst ?moneycostcnst)
(test (and(= ?appearancecnst 0) (= ?moneycostcnst 4)))
=>
(printout t crlf crlf "Thing isn't precious." crlf)
(assert (act "glearn")))
(defrule R17
(appearancecnst ?appearancecnst)
(moneycostcnst ?moneycostcnst)
(test (and(= ?appearancecnst 0) (= ?moneycostcnst 3)))
=>
(printout t crlf crlf "Thing isn't precious with  the probability of 70%." crlf)
(assert (act "glearn")))
