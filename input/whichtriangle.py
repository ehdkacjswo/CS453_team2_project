def whichtriangle(x,y,z):
    if (x == y):
        if (x == z):
            if (y == z):
                print("Equilateral triangle")
            else:
                print("Isosceles triangle")
        elif (y == z):
            print("Isosceles triangle") 
    elif (x == z):
        if (y == z):
            print ("Isosceles triangle")
    else:
        print("Scalene triangle")