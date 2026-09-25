import {
  analyzeRequestBody,
  requestBody,
  requirePost,
  sendApiError,
  type ApiRequest,
  type ApiResponse,
} from './gemini.js'

export default async function handler(request: ApiRequest, response: ApiResponse) {
  try {
    requirePost(request, response)
    const analysis = await analyzeRequestBody(requestBody(request))
    response.status(200).json({ analysis })
  } catch (error) {
    sendApiError(response, error)
  }
}
