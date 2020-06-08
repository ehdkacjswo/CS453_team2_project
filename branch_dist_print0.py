from test import ff


def ffff(fff, a, b, c):
    ttt = (lambda xxx: xxx if xxx > 0 else (lambda xxx: xxx if xxx <= 0 else
        xxx + 0)(abs(b - c)))((lambda xxx: xxx if xxx <= 0 else xxx + 0)(
        abs(a - b)))
    fff.write('{} {}\n'.format(1, ttt))
    if a == b and b == c:
        return 1
    else:
        ttt = (lambda xxx: xxx if xxx <= 0 else (lambda xxx: xxx if xxx <= 
            0 else (lambda xxx: xxx if xxx <= 0 else xxx + 0)(abs(b - c)))(
            (lambda xxx: xxx if xxx <= 0 else xxx + 0)(abs(c - a))))((lambda
            xxx: xxx if xxx <= 0 else xxx + 0)(abs(a - b)))
        fff.write('{} {}\n'.format(2, ttt))
        if a == b or c == a or b == c:
            return 2
        else:
            return 3
