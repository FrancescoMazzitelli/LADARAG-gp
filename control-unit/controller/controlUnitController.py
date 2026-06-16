from flask import request, jsonify, Response
from flask_restx import Namespace, Resource
from service.orchestrator import Orchestrator

api = Namespace("control", description="Services management and orchestration")

category_model = api.schema_model(
    "CategorySchema",
    {"type": "object", "additionalProperties": {"type": "string"},
     "example": {"input": "your text here"}},
)


@api.route("/invoke")
@api.expect(category_model)
class ConversationalAgent(Resource):
    def post(self):
        data = request.get_json(force=True)
        user_input = data["input"]
        print(f"Input ricevuto: {user_input}")

        orchestrator = Orchestrator()
        results = orchestrator.control(user_input)
        # control() puo' restituire una flask Response (caso download file)
        if isinstance(results, Response):
            return results
        return jsonify(results)
