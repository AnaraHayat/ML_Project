def get_recommendation(risk_level):

    if risk_level == "Critical":
        return {
            "action": "Immediate Help",
            "message": "Contact therapist immediately",
            "resource": "Emergency hotline / Psychiatrist"
        }

    elif risk_level == "High":
        return {
            "action": "Seek Help",
            "message": "You should talk to a therapist",
            "resource": "CBT / DBT therapist"
        }

    elif risk_level == "Medium":
        return {
            "action": "Monitor",
            "message": "Practice mindfulness",
            "resource": "Meditation / Journaling"
        }

    else:
        return {
            "action": "Healthy Routine",
            "message": "You are stable",
            "resource": "Exercise / Sleep well"
        }