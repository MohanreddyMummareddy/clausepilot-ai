import {
  askRequestBody,
  requestBody,
  requireApiRequest,
  sendApiError,
  type ApiRequest,
  type ApiResponse,
} from './gemini.js'

export default async function handler(request: ApiRequest, response: ApiResponse) {
  try {
    requireApiRequest(request, response)
    const answer = await askRequestBody(requestBody(request))
    response.status(200).json({ answer })
  } catch (error) {
    sendApiError(response, error)
  }
}
