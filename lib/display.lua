local display = {}

function display.canvas(w, h, bg)
    --- @class Canvas
    local self = {}
    self.w = w
    self.h = h
    self.bg = bg
    self.pixels = {}
    for i = 1, w * h do self.pixels[i] = bg end

    function self.clear()
        local p, b = self.pixels, self.bg
        for i = 1, self.w * self.h do p[i] = b end
    end

    function self.put_pixel(x, y, col)
        if x < 1 or x > self.w or y < 1 or y > self.h then return end
        self.pixels[(y - 1) * self.w + x] = col
    end

    --- https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    function self.draw_line(x1, y1, x2, y2, col)
        local dx = math.abs(x2 - x1)
        local sx = x1 < x2 and 1 or -1
        local dy = -math.abs(y2 - y1)
        local sy = y1 < y2 and 1 or -1
        local err = dx + dy

        local put = self.put_pixel
        while true do
            put(x1, y1, col)
            local err2 = 2 * err
            if err2 >= dy then
                if x1 == x2 then break end
                err = err + dy
                x1 = x1 + sx
            end
            if err2 <= dx then
                if y1 == y2 then break end
                err = err + dx
                y1 = y1 + sy
            end
        end
    end

    --- This is a hollow circle.
    function self.draw_circle(cx, cy, r, col)
        local x = 0
        local y = r
        local p = 1 - r
        local put = self.put_pixel

        while x <= y do
            put(cx + x, cy + y, col)
            put(cx - x, cy + y, col)
            put(cx + x, cy - y, col)
            put(cx - x, cy - y, col)
            put(cx + y, cy + x, col)
            put(cx - y, cy + x, col)
            put(cx + y, cy - x, col)
            put(cx - y, cy - x, col)

            x = x + 1
            if p < 0 then
                p = p + 2 * x + 1
            else
                y = y - 1
                p = p + 2 * (x - y) + 1
            end
        end
    end

    function self.fill(x, y, width, height, col)
        local _w, p = self.w, self.pixels
        local xs, xe = math.max(x, 1), math.min(x + width - 1, _w)
        local ys, ye = math.max(y, 1), math.min(y + height - 1, self.h)
        local y_off = (ys - 1) * _w
        for _ = ys, ye do
            for _x = xs, xe do
                p[y_off + _x] = col
            end
            y_off = y_off + _w
        end
    end

    --- @param bitmap table<boolean> `false` for transparency.
    --- @param width integer Every row must be equal in length.
    function self.blit_bitmap(x, y, col, bitmap, width)
        local height = #bitmap / width
        local _w, p = self.w, self.pixels
        local xs, xe = math.max(x, 1), math.min(x + width - 1, _w)
        local ys, ye = math.max(y, 1), math.min(y + height - 1, self.h)

        local src_off = (ys - y) * width - x + 1
        local dst_off = (ys - 1) * _w
        for _ = ys, ye do
            for _x = xs, xe do
                if bitmap[src_off + _x] then p[dst_off + _x] = col end
            end
            src_off = src_off + width
            dst_off = dst_off + _w
        end
    end

    --- @param map table<string> `" "` for transparency.
    --- @param width integer Every row must be equal in length.
    function self.blit_map(x, y, map, width)
        local height = #map / width
        local _w, p = self.w, self.pixels
        local xs, xe = math.max(x, 1), math.min(x + width - 1, _w)
        local ys, ye = math.max(y, 1), math.min(y + height - 1, self.h)

        local src_off = (ys - y) * width - x + 1
        local dst_off = (ys - 1) * _w
        for _ = ys, ye do
            for _x = xs, xe do
                local px = map[src_off + _x]
                if px ~= " " then p[dst_off + _x] = px end
            end
            src_off = src_off + width
            dst_off = dst_off + _w
        end
    end

    function self.fill_triangle(x1, y1, x2, y2, x3, y3, col)
        -- Sort by ascending y
        if y1 > y2 then x1, x2, y1, y2 = x2, x1, y2, y1 end
        if y1 > y3 then x1, x3, y1, y3 = x3, x1, y3, y1 end
        if y2 > y3 then x2, x3, y2, y3 = x3, x2, y3, y2 end

        if y3 <= y1 then return end            -- No area

        local slope_13 = (x3 - x1) / (y3 - y1) -- Long edge
        local slope_12, slope_23 = 0, 0        -- Top, bottom. 0 is fallback.
        if y2 > y1 then slope_12 = (x2 - x1) / (y2 - y1) end
        if y3 > y2 then slope_23 = (x3 - x2) / (y3 - y2) end

        local _w, _h, p = self.w, self.h, self.pixels
        local min, max, floor = math.min, math.max, math.floor
        for y = max(y1, 1), min(y3, _h) do
            local xs, xe
            if y < y2 then
                -- Top half interpolation
                local dy = y - y1
                xs = x1 + slope_13 * dy
                xe = x1 + slope_12 * dy
            else
                -- Bottom half interpolation
                xs = x1 + slope_13 * (y - y1)
                xe = x2 + slope_23 * (y - y2)
            end
            -- Ensure left to right
            if xs > xe then xs, xe = xe, xs end
            -- Scanline
            local y_off = (y - 1) * _w
            for x = max(floor(xs + 0.5), 1), min(floor(xe + 0.5), _w) do
                p[y_off + x] = col
            end
        end
    end

    --- @TODO: this is slow as fuck; improve! Also it's ugly
    --- @TODO: fix the gaps that sometimes occur. Idk why that happens.
    --- Uses affine texture mapping
    --- @param v1 table `{ x, y, u, v }` u, v must be between 0.0 and 1.0
    --- @param v2 table `{ x, y, u, v }`
    --- @param v3 table `{ x, y, u, v }`
    --- @param map table<string> `" "` for transparency.
    --- @param width integer Every row must be equal in length.
    function self.blit_triangle(v1, v2, v3, map, width)
        local X, Y, U, V = 1, 2, 3, 4

        -- Sort by ascending y
        if v1[Y] > v2[Y] then v1, v2 = v2, v1 end
        if v1[Y] > v3[Y] then v1, v3 = v3, v1 end
        if v2[Y] > v3[Y] then v2, v3 = v3, v2 end

        if v3[Y] <= v1[Y] then return end -- No area

        -- Long edge
        local idy13 = 1 / (v3[Y] - v1[Y])
        local slope_x13 = (v3[X] - v1[X]) * idy13
        local slope_u13 = (v3[U] - v1[U]) * idy13
        local slope_v13 = (v3[V] - v1[V]) * idy13
        -- Fallbacks in case both conditions fail.
        local slope_x12, slope_u12, slope_v12 = 0, 0, 0
        local slope_x23, slope_u23, slope_v23 = 0, 0, 0
        -- Top
        if v2[Y] > v1[Y] then
            local idy12 = 1 / (v2[Y] - v1[Y])
            slope_x12 = (v2[X] - v1[X]) * idy12
            slope_u12 = (v2[U] - v1[U]) * idy12
            slope_v12 = (v2[V] - v1[V]) * idy12
        end
        -- Bottom
        if v3[Y] > v2[Y] then
            local idy23 = 1 / (v3[Y] - v2[Y])
            slope_x23 = (v3[X] - v2[X]) * idy23
            slope_u23 = (v3[U] - v2[U]) * idy23
            slope_v23 = (v3[V] - v2[V]) * idy23
        end

        local height = #map / width
        local wm1, hm1 = width - 1, height - 1
        local _w, _h, p = self.w, self.h, self.pixels
        local min, max, floor = math.min, math.max, math.floor

        for y = max(v1[Y], 1), min(v3[Y], _h) do
            local xs, xe, us, ue, vs, ve
            if y < v2[Y] then
                -- Top half interpolation
                local dy1 = y - v1[Y]
                xs = v1[X] + slope_x13 * dy1
                xe = v1[X] + slope_x12 * dy1
                us = v1[U] + slope_u13 * dy1
                ue = v1[U] + slope_u12 * dy1
                vs = v1[V] + slope_v13 * dy1
                ve = v1[V] + slope_v12 * dy1
            else
                -- Bottom half interpolation
                local dy1 = y - v1[Y]
                local dy2 = y - v2[Y]
                xs = v1[X] + slope_x13 * dy1
                xe = v2[X] + slope_x23 * dy2
                us = v1[U] + slope_u13 * dy1
                ue = v2[U] + slope_u23 * dy2
                vs = v1[V] + slope_v13 * dy1
                ve = v2[V] + slope_v23 * dy2
            end
            -- Ensure left to right
            if xs > xe then
                xs, xe = xe, xs
                us, ue = ue, us
                vs, ve = ve, vs
            end

            -- Scanline prep
            local y_off = (y - 1) * _w
            local xs_r, xe_r = max(floor(xs + 0.5), 1), min(floor(xe + 0.5), _w)
            if xs_r > xe_r then goto continue end

            local dx = xe - xs
            if dx == 0 then
                -- Single pixel case
                local tx = floor(us * wm1 + 1.5)
                local ty = floor(vs * hm1 + 1.5)
                local pixel = map[(ty - 1) * width + tx]
                if pixel ~= " " then p[y_off + xs] = pixel end
                goto continue
            end

            -- Real scanline
            local i_dx = 1 / dx
            local du = (ue - us) * i_dx
            local dv = (ve - vs) * i_dx
            local dxsr = xs_r - xs
            local u = us + du * dxsr
            local v = vs + dv * dxsr
            for x = xs_r, xe_r do
                local tx = floor(u * wm1 + 1.5)
                local ty = floor(v * hm1 + 1.5)
                if tx >= 1 and tx <= width and ty >= 1 and ty <= height then
                    local pixel = map[(ty - 1) * width + tx]
                    if pixel ~= " " then p[y_off + x] = pixel end
                end
                u = u + du
                v = v + dv
            end

            ::continue::
        end
    end

    return self
end

--- Not used, but kept as I just really like it.
local function shrink_bitmap_2x3(b1, b2, b3, b4, b5, b6)
    local count =
        (b1 and 1 or 0) +
        (b2 and 1 or 0) +
        (b3 and 1 or 0) +
        (b4 and 1 or 0) +
        (b5 and 1 or 0) +
        (b6 and 1 or 0)

    if count == 0 then return " ", false end
    if count == 6 then return " ", true end

    local swap = count >= 3
    if b6 ~= swap then swap = not swap end -- Special case for the last bit

    local ch = 128
    if b1 ~= swap then ch = ch + 1 end
    if b2 ~= swap then ch = ch + 2 end
    if b3 ~= swap then ch = ch + 4 end
    if b4 ~= swap then ch = ch + 8 end
    if b5 ~= swap then ch = ch + 16 end
    return string.char(ch), swap
end

--- I was **this** close to brute-force mapping all 64 combinations by hand.
--- My saviour: https://github.com/exerro/ccgl/blob/master/src/functions/texture_subpixel_convert.lua
local function shrink_pixels_2x3(c1, c2, c3, c4, c5, c6)
    if c1 == c2 and c2 == c3 and c3 == c4 and c4 == c5 and c5 == c6 then
        return " ", c1, c1
    end

    local colour_counts = {}
    local colours = { c1, c2, c3, c4, c5, c6 }
    for i = 1, 6 do
        local col = colours[i]
        colour_counts[col] = (colour_counts[col] or 0) + 1
    end

    local major_col, minor_col
    local major_count, minor_count = 0, 0
    for col, count in pairs(colour_counts) do
        if count > major_count then
            minor_col, minor_count = major_col, major_count
            major_col, major_count = col, count
        elseif count > minor_count then
            minor_col, minor_count = col, count
        end
    end

    if major_count >= 3 then major_col, minor_col = minor_col, major_col end
    if c6 ~= minor_col then major_col, minor_col = minor_col, major_col end

    local ch = 128
    if c1 ~= minor_col then ch = ch + 1 end
    if c2 ~= minor_col then ch = ch + 2 end
    if c3 ~= minor_col then ch = ch + 4 end
    if c4 ~= minor_col then ch = ch + 8 end
    if c5 ~= minor_col then ch = ch + 16 end
    return string.char(ch), major_col, minor_col
end

--- @param cv Canvas
--- @param x integer?
--- @param y integer?
function display.blit_canvas(win, cv, x, y)
    assert(cv.w % 2 == 0, cv.w .. " not multiple of 2.")
    assert(cv.h % 3 == 0, cv.h .. " not multiple of 3.")

    -- Location of canvas, from top-left corner.
    x = x and x or 1
    y = y and y or 1

    -- Actual width, height
    local aw, ah = cv.w / 2, cv.h / 3
    local cvw, cvp = cv.w, cv.pixels

    for _y = 1, ah do
        local chrs = {}
        local tcs = {}
        local bgcs = {}

        for _x = 1, aw do
            local i = ((_y - 1) * 3 * cvw) + ((_x - 1) * 2) + 1

            local c1 = cvp[i]                 -- top-left
            local c2 = cvp[i + 1]             -- top-right
            local c3 = cvp[i + cvw]           -- middle-left
            local c4 = cvp[i + cvw + 1]       -- middle-right
            local c5 = cvp[i + (2 * cvw)]     -- bottom-left
            local c6 = cvp[i + (2 * cvw) + 1] -- bottom-right

            chrs[_x], tcs[_x], bgcs[_x] = shrink_pixels_2x3(c1, c2, c3, c4, c5, c6)
        end
        -- Do NOT blit separately for every pixel; it will cause massive stutters!
        win.setCursorPos(x, y + _y - 1)
        win.blit(table.concat(chrs), table.concat(tcs), table.concat(bgcs))
    end
end

return display
