from copy import deepcopy
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from litellm.types.llms.anthropic import AllAnthropicMessageValues
from litellm.types.llms.bedrock import (
    RequestObject,
    CommonRequestObject,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from datetime import datetime, timedelta
import json
import hashlib
import httpx

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter
)

from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)

from litellm.litellm_core_utils.prompt_templates.factory import (
    _bedrock_converse_messages_pt,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.secret_managers.main import get_secret
from ... import sap_token_cache
from ...chat.converse_transformation import SAPConverseConfig
from ...common_utils import SAPOAuthToken
from ....openai_like.common_utils import OpenAILikeError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SAPAnthropicClaudeMessagesConverseConfig(
    AnthropicMessagesConfig, SAPConverseConfig
):
    """
    Transform Anthropic Messages format to SAP AI Core Converse API format for Claude models.

    This implementation:
    - Converts Anthropic /v1/messages requests to SAP AI Core converse API format
    - Handles SAP OAuth 2.0 client credentials flow with token caching
    - Transforms responses from SAP's Bedrock-compatible format back to Anthropic format
    - Supports streaming via Server-Sent Events (SSE) with custom parsing
    - Maps usage metrics between SAP format (inputTokens/outputTokens) and Anthropic format
    - Handles all content types: text, tool_use, and reasoning/thinking blocks

    Inherits from AnthropicMessagesConfig for base message handling and SAPConverseConfig
    for SAP-specific authentication and URL construction.
    """

    DEFAULT_SAP_ANTHROPIC_API_VERSION = "bedrock-2023-05-31"

    def __init__(self, **kwargs):
        BaseAnthropicMessagesConfig.__init__(self, **kwargs)
        self.token_cache = sap_token_cache

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "sap_ai_core"

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """
        Override to add SAP OAuth token support for converse API.
        If SAP credentials are provided, use OAuth token instead of API key.
        """
        # Try to get SAP token first
        sap_token = self._get_sap_token_from_params(optional_params, headers)

        if sap_token:
            if api_base is None:
                # Get SAP AI Core base URL
                sap_base_url = get_secret("SAP_AI_CORE_BASE_URL")
                if sap_base_url and isinstance(sap_base_url, str):
                    api_base = sap_base_url
                else:
                    raise OpenAILikeError(
                        status_code=400,
                        message="Missing SAP AI Core API Base URL - Please set SAP_AI_CORE_BASE_URL",
                    )

            # Set up headers with OAuth token
            headers.update({
                "Content-Type": "application/json",
                "Authorization": f"{sap_token.token_type} {sap_token.access_token}",
                "AI-Resource-Group": "default",
            })

        return headers, api_base

    def _get_sap_token_from_params(
        self,
        optional_params: dict,
        headers: Optional[dict] = None,
    ) -> Optional[SAPOAuthToken]:
        """
        Extract SAP credentials from optional_params and get OAuth token.
        Returns None if SAP credentials are not provided.
        """
        # Check if SAP credentials are provided
        sap_client_id = optional_params.pop("sap_client_id", None) or get_secret("UAA_CLIENT_ID")
        sap_client_secret = optional_params.pop("sap_client_secret", None) or get_secret("UAA_CLIENT_SECRET")
        sap_xsuaa_url = optional_params.pop("sap_xsuaa_url", None) or get_secret("UAA_URL")

        # If no SAP credentials, return None (fallback to regular auth)
        if not all([sap_client_id, sap_client_secret, sap_xsuaa_url]):
            return None

        # Type check credentials
        if not (isinstance(sap_client_id, str) and isinstance(sap_client_secret, str) and isinstance(sap_xsuaa_url, str)):
            return None

        # Check cache
        cache_key = self._get_cache_key(sap_client_id, sap_client_secret, sap_xsuaa_url)
        cached_token = self.token_cache.get_cache(cache_key)

        if cached_token and isinstance(cached_token, SAPOAuthToken):
            if cached_token.expires_at > datetime.now():
                return cached_token

        # Get new token
        token = self._get_sap_oauth_token(
            client_id=sap_client_id,
            client_secret=sap_client_secret,
            xsuaa_url=sap_xsuaa_url,
        )

        # Cache the token
        ttl = (token.expires_at - datetime.now()).total_seconds()
        self.token_cache.set_cache(cache_key, token, ttl=int(ttl))

        return token

    def _get_cache_key(self, client_id: str, client_secret: str, xsuaa_url: str) -> str:
        """Generate a unique cache key based on credentials"""
        credential_str = json.dumps({
            "client_id": client_id,
            "client_secret": client_secret,
            "xsuaa_url": xsuaa_url
        }, sort_keys=True)
        return f"sap_oauth_{hashlib.sha256(credential_str.encode()).hexdigest()}"

    def _get_sap_oauth_token(
        self,
        client_id: str,
        client_secret: str,
        xsuaa_url: str,
    ) -> SAPOAuthToken:
        """
        Exchange client credentials for an OAuth token via SAP xsuaa.
        """
        token_endpoint = f"{xsuaa_url}/oauth/token"

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }

        try:
            response = httpx.post(
                token_endpoint,
                headers=headers,
                data=data,
                timeout=30.0,
            )
            response.raise_for_status()

            token_data = response.json()

            # Calculate token expiration time (subtract 60 seconds for safety margin)
            expires_in = token_data.get("expires_in", 3600)
            expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

            return SAPOAuthToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=expires_in,
                expires_at=expires_at,
            )

        except httpx.HTTPStatusError as e:
            raise OpenAILikeError(
                status_code=e.response.status_code,
                message=f"Failed to get SAP OAuth token: {e.response.text}",
            )
        except Exception as e:
            raise OpenAILikeError(
                status_code=500,
                message=f"Error getting SAP OAuth token: {str(e)}",
            )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Construct the complete URL for SAP AI Core converse deployment.
        Supports both regular and streaming endpoints.
        """
        # Extract deployment ID if provided
        deployment_id = optional_params.get("sap_deployment_id")
        if not deployment_id:
            raise OpenAILikeError(
                status_code=400,
                message="Missing SAP deployment ID - Please provide sap_deployment_id parameter",
            )

        # For SAP AI Core converse, use the appropriate endpoint based on streaming
        if stream is True:
            complete_url = f"{api_base}/v2/inference/deployments/{deployment_id}/converse-stream"
        else:
            complete_url = f"{api_base}/v2/inference/deployments/{deployment_id}/converse"
        return complete_url

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[AllAnthropicMessageValues],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> RequestObject:
        """
        Transform Anthropic messages format to SAP converse format.
        
        This converts Anthropic messages to SAP's converse API format (similar to how Bedrock converse works).
        """

        # Start with the base Anthropic transformation
        anthropic_messages_request = AnthropicMessagesConfig.transform_anthropic_messages_request(
            self=self,
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        request_openai_format = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(anthropic_message_request=anthropic_messages_request)
        optional_params_openai_format = deepcopy(anthropic_messages_optional_request_params)
        optional_params_openai_format["tools"] = request_openai_format.get("tools", [])

        non_system_messages, _ = self._transform_system_message(
            request_openai_format.get("messages"))

        _data: CommonRequestObject = self._transform_request_helper(model=model,
                                       system_content_blocks=anthropic_messages_optional_request_params.get("system", []),
                                       optional_params=optional_params_openai_format,
                                       messages=non_system_messages )

        bedrock_messages = (
            _bedrock_converse_messages_pt(
                messages=non_system_messages,
                model=model,
                llm_provider="bedrock_converse",
                # user_continue_message=litellm_params.pop("user_continue_message", None),
            ))

        data: RequestObject = {"messages": bedrock_messages, **_data}

        return data

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        """
        Transform Bedrock response format to Anthropic message format.

        Converts SAP AI Core Bedrock-style response to Anthropic's /v1/messages format.
        """
        
        try:
            bedrock_response = raw_response.json()
        except Exception as e:
            raise OpenAILikeError(
                status_code=422,
                message=f"Error parsing SAP response: {str(e)}",
            )

        # Extract message content from Bedrock format
        message_content = bedrock_response.get("output", {}).get("message", {})
        content_blocks = message_content.get("content", [])

        # Transform content blocks to Anthropic format
        anthropic_content = []

        for block in content_blocks:
            # Handle reasoning content (thinking)
            if "reasoningContent" in block:
                reasoning_block = block["reasoningContent"]
                if "reasoningText" in reasoning_block:
                    thinking_block = {
                        "type": "thinking",
                        "thinking": reasoning_block["reasoningText"]["text"]
                    }
                    # Add signature if present
                    if "signature" in reasoning_block["reasoningText"]:
                        thinking_block["signature"] = reasoning_block["reasoningText"]["signature"]

                    anthropic_content.append(thinking_block)

            # Handle text content
            elif "text" in block:
                anthropic_content.append({
                    "type": "text",
                    "text": block["text"]
                })

            # Handle tool use
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                anthropic_content.append({
                    "type": "tool_use",
                    "id": tool_use["toolUseId"],
                    "name": tool_use["name"],
                    "input": tool_use["input"]
                })

        # Transform usage information
        bedrock_usage = bedrock_response.get("usage", {})
        anthropic_usage = {
            "input_tokens": bedrock_usage.get("inputTokens", 0),
            "output_tokens": bedrock_usage.get("outputTokens", 0)
        }

        # Add cache-related tokens if present
        if "cacheReadInputTokens" in bedrock_usage:
            anthropic_usage["cache_read_input_tokens"] = bedrock_usage["cacheReadInputTokens"]
        if "cacheWriteInputTokens" in bedrock_usage:
            anthropic_usage["cache_creation_input_tokens"] = bedrock_usage["cacheWriteInputTokens"]

        # Handle cache creation details if present
        if bedrock_usage.get("cacheReadInputTokens", 0) > 0 or bedrock_usage.get("cacheWriteInputTokens", 0) > 0:
            anthropic_usage["cache_creation"] = {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 0
            }

        # Determine service tier (default to standard)
        service_tier = "standard"

        # Build Anthropic response format
        anthropic_response = {
            "id": f"msg_{abs(hash(str(bedrock_response)))}"[:25],  # Generate consistent ID
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": anthropic_content,
            "stop_reason": self._map_stop_reason(bedrock_response.get("stopReason", "end_turn")),
            "stop_sequence": None,
            "usage": anthropic_usage
        }

        # Add service tier if cache tokens present
        if "cache_read_input_tokens" in anthropic_usage or "cache_creation_input_tokens" in anthropic_usage:
            anthropic_response["usage"]["service_tier"] = service_tier

        return AnthropicMessagesResponse(**anthropic_response)

    def _map_stop_reason(self, bedrock_stop_reason: str) -> str:
        """Map Bedrock stop reasons to Anthropic format."""
        stop_reason_mapping = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
            "stop_sequence": "stop_sequence"
        }
        return stop_reason_mapping.get(bedrock_stop_reason, "end_turn")

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        """
        Handle streaming responses for SAP converse API.
        SAP uses SSE (Server-Sent Events) format, not AWS binary streams.
        """
        # Create SAP-specific streaming iterator to handle usage metrics transformation
        sap_streaming_iterator = SAPAnthropicMessagesStreamingIterator(
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )
        
        # Parse SSE stream and transform usage metrics
        completion_stream = sap_streaming_iterator.parse_sse_stream(
            httpx_response.aiter_bytes(chunk_size=1024)
        )
        
        # Convert to SSE format expected by Anthropic clients using base class method
        return sap_streaming_iterator.async_sse_wrapper(completion_stream)

class SAPAnthropicMessagesStreamingIterator:
    """
    SAP-specific streaming iterator that handles SSE parsing and usage metrics transformation.
    SAP returns usage metrics in a different format than Anthropic expects.
    """
    
    def __init__(
        self,
        litellm_logging_obj: LiteLLMLoggingObj,
        request_body: dict,
    ):
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            BaseAnthropicMessagesStreamingIterator,
        )
        
        self.base_iterator = BaseAnthropicMessagesStreamingIterator(
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )
    
    async def parse_sse_stream(self, byte_stream) -> AsyncIterator[dict]:
        """
        Parse SAP's SSE stream and transform usage metrics format.
        
        SAP returns: {'metadata': {'usage': {'inputTokens': 8, 'outputTokens': 20}}}
        Anthropic expects: {'usage': {'input_tokens': 8, 'output_tokens': 20}}
        """
        import json
        
        buffer = b""
        async for chunk in byte_stream:
            buffer += chunk
            
            # Process complete SSE events
            while b"\n\n" in buffer:
                event_data, buffer = buffer.split(b"\n\n", 1)
                event_str = event_data.decode('utf-8')
                
                # Parse SSE format: "data: {...}"
                for line in event_str.split('\n'):
                    line = line.strip()
                    if line.startswith('data: '):
                        try:
                            # Extract JSON data after "data: "
                            json_str = line[6:]  # Remove "data: " prefix
                            # Handle malformed JSON by using eval for dict-like strings
                            if json_str.startswith('{') and json_str.endswith('}'):
                                try:
                                    data = json.loads(json_str)
                                except json.JSONDecodeError:
                                    # Fallback for dict-like strings: {'key': 'value'}
                                    data = eval(json_str)
                                
                                # Transform usage metrics if present
                                data = self._transform_usage_metrics(data)
                                yield data
                                
                        except (json.JSONDecodeError, SyntaxError, ValueError) as e:
                            # Skip malformed data
                            continue
    
    def _transform_usage_metrics(self, data: dict) -> dict:
        """
        Transform SAP usage metrics format to Anthropic format.
        
        SAP format: {'metadata': {'usage': {'inputTokens': 8, 'outputTokens': 20, 'totalTokens': 28}}}
        Anthropic format: {'usage': {'input_tokens': 8, 'output_tokens': 20}}
        """
        if 'metadata' in data:
            metadata = data.pop('metadata')
            if 'usage' in metadata:
                sap_usage = metadata['usage']
                anthropic_usage = {}
                
                # Transform field names
                if 'inputTokens' in sap_usage:
                    anthropic_usage['input_tokens'] = sap_usage['inputTokens']
                if 'outputTokens' in sap_usage:
                    anthropic_usage['output_tokens'] = sap_usage['outputTokens']
                # Note: totalTokens is not part of Anthropic spec, so we omit it
                
                if anthropic_usage:
                    data['usage'] = anthropic_usage
        
        return data
    
    def _convert_chunk_to_sse_format(self, chunk: Union[dict, Any]) -> bytes:
        """Transform SAP format to Anthropic SSE format."""
        if not isinstance(chunk, dict):
            return self.base_iterator._convert_chunk_to_sse_format(chunk)
        
        # Transform SAP format to Anthropic format
        transformed_chunk = self._transform_sap_to_anthropic_format(chunk)
        
        # Use base iterator's SSE formatting with transformed data
        return self.base_iterator._convert_chunk_to_sse_format(transformed_chunk)
    
    def _transform_sap_to_anthropic_format(self, sap_chunk: dict) -> dict:
        """Transform SAP chunk format to Anthropic format."""
        # Handle messageStart -> message_start
        if 'messageStart' in sap_chunk:
            return {
                "type": "message_start",
                "message": {
                    "id": f"msg_{abs(hash(str(sap_chunk)))}",
                    "type": "message",
                    "role": sap_chunk['messageStart']['role'],
                    "content": [],
                    "model": "claude-sonnet-4",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            }
        
        # Handle contentBlockStart -> content_block_start
        elif 'contentBlockStart' in sap_chunk:
            start_data = sap_chunk['contentBlockStart']['start']
            content_block_index = sap_chunk['contentBlockStart']['contentBlockIndex']
            
            # Handle tool use start
            if 'toolUse' in start_data:
                tool_use_data = start_data['toolUse']
                return {
                    "type": "content_block_start",
                    "index": content_block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_use_data.get('toolUseId', ''),
                        "name": tool_use_data.get('name', ''),
                        "input": {}
                    }
                }
            # Handle text start
            elif 'text' in start_data:
                return {
                    "type": "content_block_start",
                    "index": content_block_index,
                    "content_block": {
                        "type": "text",
                        "text": ""
                    }
                }
            # Handle thinking/reasoning start
            elif 'reasoningContent' in start_data:
                return {
                    "type": "content_block_start",
                    "index": content_block_index,
                    "content_block": {
                        "type": "thinking",
                        "thinking": ""
                    }
                }
            # Fallback - assume text
            else:
                return {
                    "type": "content_block_start", 
                    "index": content_block_index,
                    "content_block": {
                        "type": "text",
                        "text": ""
                    }
                }
        
        # Handle contentBlockDelta -> content_block_delta
        elif 'contentBlockDelta' in sap_chunk:
            delta = sap_chunk['contentBlockDelta']['delta']
            content_block_index = sap_chunk['contentBlockDelta']['contentBlockIndex']
            
            # Handle tool use input delta
            if 'toolUse' in delta and 'input' in delta['toolUse']:
                return {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": delta['toolUse']['input']
                    }
                }
            # Handle reasoning content (thinking)
            elif 'reasoningContent' in delta:
                return {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": delta['reasoningContent'].get('text', '')
                    }
                }
            # Handle regular text delta
            elif 'text' in delta:
                return {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {
                        "type": "text_delta",
                        "text": delta['text']
                    }
                }
        
        # Handle contentBlockStop -> content_block_stop
        elif 'contentBlockStop' in sap_chunk:
            return {
                "type": "content_block_stop",
                "index": sap_chunk['contentBlockStop']['contentBlockIndex']
            }
        
        # Handle messageStop -> message_stop
        elif 'messageStop' in sap_chunk:
            return {
                "type": "message_stop"
            }
        
        # Handle metadata -> message_delta (for usage)
        elif 'metadata' in sap_chunk:
            transformed_usage = self._transform_usage_metrics(sap_chunk.copy())
            return {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn"
                },
                "usage": transformed_usage.get('usage', {})
            }
        
        # Return as-is if no transformation needed
        return sap_chunk
    
    async def async_sse_wrapper(self, completion_stream) -> AsyncIterator[bytes]:
        """Wrap the completion stream with SSE format using custom chunk transformation."""
        collected_chunks = []
        
        async for chunk in completion_stream:
            encoded_chunk = self._convert_chunk_to_sse_format(chunk)
            collected_chunks.append(encoded_chunk)
            yield encoded_chunk
        
        # Handle logging after all chunks are processed
        await self.base_iterator._handle_streaming_logging(collected_chunks)

