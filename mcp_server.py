import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from supabase import create_client, Client
from dotenv import load_dotenv
import io # Added for BytesIO

from PIL import Image, ImageDraw, ImageFont # Added for image manipulation

# Load environment variables
load_dotenv()

@dataclass
class AppContext:
    supabase: Client
    user_id: str

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with Supabase connection"""
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    user_id = os.getenv("USER_ID")
    
    if not all([supabase_url, supabase_key, user_id]):
        raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_ANON_KEY, USER_ID")
    
    supabase = create_client(supabase_url, supabase_key)
    
    try:
        yield AppContext(supabase=supabase, user_id=user_id)
    finally:
        # Cleanup if needed
        pass

# Create MCP server
mcp = FastMCP(
    "Alzheimer Assistant",
    dependencies=["supabase", "python-dotenv", "Pillow"], # Added Pillow
    lifespan=app_lifespan
)

@mcp.tool()
def add_reminder(
    title: str,
    description: str,
    reminder_time: str,
    daily_recurring: bool = False
) -> str:
    """Add a new reminder for the user.
    
    Args:
        title: Title of the reminder
        description: Description of the reminder
        reminder_time: When to remind (ISO format: YYYY-MM-DDTHH:MM:SS)
        daily_recurring: Whether this reminder repeats daily
    """
    try:
        ctx = mcp.get_context()
        supabase = ctx.request_context.lifespan_context.supabase
        user_id = ctx.request_context.lifespan_context.user_id
        
        # Parse datetime
        try:
            parsed_time = datetime.fromisoformat(reminder_time.replace('Z', '+00:00'))
        except ValueError:
            return f"Error: Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
        
        # Insert reminder
        result = supabase.table("reminders").insert({
            "user_id": user_id,
            "title": title,
            "description": description,
            "reminder_time": parsed_time.isoformat(),
            "daily_recurring": daily_recurring
        }).execute()
        
        if result.data:
            reminder_id = result.data[0]["id"]
            return f"Reminder created successfully with ID: {reminder_id}"
        else:
            # Assuming result.error might contain more info if available from client
            error_info = getattr(result, 'error', 'Unknown error')
            return f"Error: Failed to create reminder. Details: {error_info}"
            
    except Exception as e:
        return f"Error adding reminder: {str(e)}"

@mcp.tool()
def get_all_reminders() -> str:
    """Get all reminders for the user."""
    try:
        ctx = mcp.get_context()
        supabase = ctx.request_context.lifespan_context.supabase
        user_id = ctx.request_context.lifespan_context.user_id
        
        result = supabase.table("reminders").select("*").eq("user_id", user_id).order("reminder_time").execute()
        
        if not result.data:
            return "No reminders found."
        
        reminders_text = "Your Reminders:\n\n"
        for reminder in result.data:
            created_at = datetime.fromisoformat(reminder["created_at"].replace('Z', '+00:00'))
            reminder_time = datetime.fromisoformat(reminder["reminder_time"].replace('Z', '+00:00'))
            recurring = "Daily" if reminder["daily_recurring"] else "One-time"
            
            reminders_text += f"ID: {reminder['id']}\n"
            reminders_text += f"Title: {reminder['title']}\n"
            reminders_text += f"Description: {reminder['description']}\n"
            reminders_text += f"Reminder Time: {reminder_time.strftime('%Y-%m-%d %H:%M')}\n"
            reminders_text += f"Type: {recurring}\n"
            reminders_text += f"Created: {created_at.strftime('%Y-%m-%d %H:%M')}\n"
            reminders_text += "-" * 50 + "\n"
        
        return reminders_text
        
    except Exception as e:
        return f"Error getting reminders: {str(e)}"

@mcp.tool()
def remove_reminder(reminder_id: str) -> str:
    """Remove a reminder by ID.
    
    Args:
        reminder_id: The ID of the reminder to remove
    """
    try:
        ctx = mcp.get_context()
        supabase = ctx.request_context.lifespan_context.supabase
        user_id = ctx.request_context.lifespan_context.user_id
        
        # First check if reminder exists and belongs to user
        check_result = supabase.table("reminders").select("id").eq("id", reminder_id).eq("user_id", user_id).execute()
        
        if not check_result.data:
            return f"Error: Reminder with ID {reminder_id} not found or doesn't belong to you."
        
        # Delete the reminder
        result = supabase.table("reminders").delete().eq("id", reminder_id).eq("user_id", user_id).execute()
        
        if result.data:
            return f"Reminder {reminder_id} removed successfully."
        else:
            error_info = getattr(result, 'error', 'Unknown error')
            return f"Error: Failed to remove reminder {reminder_id}. Details: {error_info}"
            
    except Exception as e:
        return f"Error removing reminder: {str(e)}"

@mcp.tool()
def add_memory(
    title: str,
    content: str,
    image_description: Optional[str] = None,
    add_image: bool = False
) -> str:
    """Add a new memory for the user.
    
    Args:
        title: Title of the memory
        content: Content/description of the memory
        image_description: Optional. If provided and add_image is True, 
                           this text will be annotated on the image. 
                           The image is expected to be 640x480. This is compulsory if add_image is True. Do not use any special characters here.
        add_image: Whether to include an image from 'temp_image.jpg'
        
    """
    try:
        ctx = mcp.get_context()
        supabase = ctx.request_context.lifespan_context.supabase
        user_id = ctx.request_context.lifespan_context.user_id
        
        image_url: Optional[str] = None
        
        if add_image:
            input_image_path = "temp_image.jpg"
            if not os.path.exists(input_image_path):
                return "Error: temp_image.jpg not found, but add_image was true."

            try:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                base_filename = os.path.splitext(os.path.basename(input_image_path))[0]
                
                upload_file_data: bytes # This will now consistently hold bytes
                storage_filename: str
                content_type_for_upload: str

                if image_description:
                    # Annotate the image
                    img = Image.open(input_image_path).convert("RGBA")
                    
                    if img.size != (640, 480):
                        img = img.resize((640, 480))
                    
                    draw = ImageDraw.Draw(img)
                    font_size = 50
                    font_name = "arial.ttf"
                    font = None
                    try:
                        font = ImageFont.truetype(font_name, font_size)
                    except IOError:
                        font = ImageFont.load_default()

                    if hasattr(font, "getmetrics"): 
                        current_text_bbox = draw.textbbox((0, 0), image_description, font=font)
                        text_w = current_text_bbox[2] - current_text_bbox[0]
                        while text_w >= img.width * 0.90 and font_size > 10:
                            font_size -= 2
                            try:
                                font = ImageFont.truetype(font_name, font_size)
                                current_text_bbox = draw.textbbox((0, 0), image_description, font=font)
                                text_w = current_text_bbox[2] - current_text_bbox[0]
                            except IOError:
                                font = ImageFont.load_default()
                                break 
                    
                    final_text_bbox = draw.textbbox((0,0), image_description, font=font)
                    text_width = final_text_bbox[2] - final_text_bbox[0]
                    text_height = final_text_bbox[3] - final_text_bbox[1]
                    
                    x_pos = (img.width - text_width) / 2
                    y_pos = img.height - text_height - 10

                    text_color = (255, 255, 255, 128)
                    draw.text((x_pos, y_pos), image_description, font=font, fill=text_color)
                    
                    output_buffer = io.BytesIO()
                    img.save(output_buffer, format="PNG")
                    
                    # --- FIX 1: Get the raw bytes from the BytesIO object ---
                    upload_file_data = output_buffer.getvalue()
                    
                    storage_filename = f"{timestamp}_{base_filename}.png"
                    content_type_for_upload = "image/png"
                
                else: # No image_description, upload original image
                    # --- FIX 2: Read the file content into bytes ---
                    with open(input_image_path, "rb") as f:
                        upload_file_data = f.read()

                    storage_filename = f"{timestamp}_{os.path.basename(input_image_path)}"
                    content_type_for_upload = "image/jpeg" 

                # Perform upload
                storage_path = f"{user_id}/memories/{storage_filename}"
                
                supabase.storage.from_("memory-images").upload(
                    path=storage_path,
                    file=upload_file_data, # This is now 'bytes', which is a valid type
                    file_options={"content-type": content_type_for_upload}
                )
                
                # --- FIX 3: The manual file.close() is no longer needed ---
                # The 'with open...' handles closing automatically, and BytesIO doesn't need it here.

                public_url_response = supabase.storage.from_("memory-images").get_public_url(storage_path)
                image_url = public_url_response

            except Exception as e:
                # No file closing needed here anymore, it's handled by 'with' or not applicable
                return f"Error processing or uploading image: {str(e)}"

        # Prepare data for insertion into the database
        memory_data: Dict[str, Any] = {
            "user_id": user_id,
            "title": title,
            "content": content
        }
        if image_url:
            memory_data["image_url"] = image_url
            
        # Insert memory record into Supabase table
        result = supabase.table("memories").insert(memory_data).execute()
        
        if result.data:
            memory_id = result.data[0]["id"]
            response_message = f"Memory saved successfully with ID: {memory_id}"
            if image_url:
                response_message += f". Image available at: {image_url}"
            return response_message
        else:
            error_info = getattr(result, 'error', 'Unknown error')
            return f"Error: Failed to save memory. Details: {error_info}"
            
    except Exception as e:
        return f"Error adding memory: {str(e)}"
@mcp.tool()
def get_all_memories() -> str:
    """Get all memories for the user."""
    try:
        ctx = mcp.get_context()
        supabase = ctx.request_context.lifespan_context.supabase
        user_id = ctx.request_context.lifespan_context.user_id
        
        result = supabase.table("memories").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not result.data:
            return "No memories found."
        
        memories_text = "Your Memories:\n\n"
        for memory in result.data:
            created_at = datetime.fromisoformat(memory["created_at"].replace('Z', '+00:00'))
            
            memories_text += f"ID: {memory['id']}\n"
            memories_text += f"Title: {memory['title']}\n"
            memories_text += f"Content: {memory['content']}\n"
            
            image_url = memory.get("image_url")
            if image_url:
                memories_text += f"Image: {image_url}\n"
                memories_text += f"Image Available: Yes\n"
            else:
                memories_text += f"Image Available: No\n"
                
            memories_text += f"Created: {created_at.strftime('%Y-%m-%d %H:%M')}\n"
            memories_text += "-" * 50 + "\n"
        
        return memories_text
        
    except Exception as e:
        return f"Error getting memories: {str(e)}"

@mcp.tool()
def update_fall_detection_status(enabled: bool) -> str:
    """
    Enable or disable the fall detection feature on the device.
    
    Args:
        enabled: Set to True to enable fall detection, False to disable.
    """
    try:
        ctx = mcp.get_context()
        supabase = ctx.request_context.lifespan_context.supabase
        user_id = ctx.request_context.lifespan_context.user_id

        # Upsert ensures the record is created if it doesn't exist, or updated if it does.
        result = supabase.table("device_status").upsert({
            "user_id": user_id,
            "fall_detection_enabled": enabled,
            "updated_at": datetime.now().isoformat()
        }).execute()

        if result.data:
            status = "enabled" if enabled else "disabled"
            return f"Fall detection has been successfully {status}."
        else:
            error_info = getattr(result, 'error', 'Unknown error')
            return f"Error updating fall detection status: {error_info}"

    except Exception as e:
        return f"A server error occurred while updating status: {str(e)}"


if __name__ == "__main__":
    mcp.run()